"""
Generic tabular model Streamlit app.

실험 registry, run artifact, 모델 파일, 추론 로그를 한 화면에서 확인합니다.
실행:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import html
import pickle
import re
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yaml


BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.utils.logger import InferenceLogger  # noqa: E402


REGISTRY_PATH = BASE_DIR / "model_registry.json"
LOG_PATH = BASE_DIR / "logs" / "inference.jsonl"
MISSING_OPTION = "(missing)"
LOWER_IS_BETTER_METRICS = {"mae", "rmse"}
DISPLAY_ACRONYMS = {"ai", "api", "auc", "gbm", "id", "ml", "rf", "svm", "xg", "xt"}


def short_id(value: Any, prefix: int = 10, suffix: int = 6) -> str:
    text = str(value or "")
    if len(text) <= prefix + suffix + 3:
        return text
    return f"{text[:prefix]}...{text[-suffix:]}"


def truncate_text(value: Any, max_chars: int = 80) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars - 1]}…"


def humanize_identifier(value: Any, uppercase_short_tokens: bool = False) -> str:
    text = str(value or "").strip()
    if not text:
        return "n/a"
    text = re.sub(r"[-_]+", " ", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"\bGBMXT\b", "GBM XT", text)
    words = [word for word in text.split() if word]
    display_words = []
    for word in words:
        if uppercase_short_tokens and len(word) <= 3:
            display_words.append(word.upper())
        elif word.lower() in DISPLAY_ACRONYMS:
            display_words.append(word.upper())
        else:
            display_words.append(word[:1].upper() + word[1:])
    return " ".join(display_words)


def display_data_name(data_version: Any) -> str:
    return humanize_identifier(data_version, uppercase_short_tokens=True)


def display_experiment_type(record: dict[str, Any] | None) -> str:
    value = str((record or {}).get("experiment_type") or "baseline")
    return "AutoML" if value.lower() == "automl" else humanize_identifier(value)


def display_model_name(record: dict[str, Any] | None, selected_model: str | None = None) -> str:
    raw_name = selected_model or (record or {}).get("model_name") or (record or {}).get("model_id")
    return humanize_identifier(raw_name)


def display_created_at(record: dict[str, Any] | None) -> str:
    created_at = str((record or {}).get("created_at") or "")
    if not created_at:
        return "n/a"
    return created_at.replace("T", " ")[:16]


def display_run_label(record: dict[str, Any]) -> str:
    return f"{display_model_name(record)} · {display_experiment_type(record)} · {display_created_at(record)}"


def format_metric(value: Any) -> str:
    if isinstance(value, (int, float)) and pd.notna(value):
        return f"{value:.4f}"
    return "n/a"


def format_delta(value: Any) -> str:
    if isinstance(value, (int, float)) and pd.notna(value):
        return f"{value:+.4f}"
    return "n/a"


def format_percent(value: Any) -> str:
    if isinstance(value, (int, float)) and pd.notna(value):
        return f"{value:.1%}"
    return "n/a"


def record_key(record: dict[str, Any] | None) -> tuple[str, str]:
    if not record:
        return ("", "")
    return (str(record.get("run_id") or ""), str(record.get("model_id") or ""))


def same_record(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    return bool(left and right and record_key(left) == record_key(right))


def is_automl_record(record: dict[str, Any] | None) -> bool:
    if not record:
        return False
    return record.get("experiment_type") == "automl" or record.get("backend") == "autogluon"


def validation_metric_value(record: dict[str, Any], metric_name: str) -> float | None:
    value = record.get("metrics", {}).get(metric_name)
    if isinstance(value, (int, float)) and pd.notna(value):
        return float(value)
    return None


def metric_lower_is_better(metric_name: str) -> bool:
    return metric_name in LOWER_IS_BETTER_METRICS


def records_for_data_version(registry: list[dict[str, Any]], data_version: str | None) -> list[dict[str, Any]]:
    if data_version is None:
        return list(registry)
    return [record for record in registry if record.get("data_version") == data_version]


def data_versions_from_registry(registry: list[dict[str, Any]]) -> list[str | None]:
    versions: list[str | None] = []
    for record in registry:
        version = record.get("data_version")
        if version not in versions:
            versions.append(version)
    return versions or [None]


def ranked_records(registry: list[dict[str, Any]], primary_metric: str) -> list[dict[str, Any]]:
    with_metric = [
        record for record in registry if validation_metric_value(record, primary_metric) is not None
    ]
    without_metric = [
        record for record in registry if validation_metric_value(record, primary_metric) is None
    ]
    with_metric.sort(
        key=lambda record: (
            validation_metric_value(record, primary_metric) or float("-inf"),
            str(record.get("created_at") or ""),
        ),
        reverse=not metric_lower_is_better(primary_metric),
    )
    without_metric.sort(key=lambda record: str(record.get("created_at") or ""), reverse=True)
    return with_metric + without_metric


def best_model_record(
    registry: list[dict[str, Any]], data_version: str | None, primary_metric: str
) -> dict[str, Any] | None:
    candidates = records_for_data_version(registry, data_version)
    ranked = ranked_records(candidates, primary_metric)
    return ranked[0] if ranked and validation_metric_value(ranked[0], primary_metric) is not None else None


def baseline_record(registry: list[dict[str, Any]]) -> dict[str, Any] | None:
    ordered = sorted(registry, key=lambda record: str(record.get("created_at") or ""))
    for record in ordered:
        searchable = " ".join(
            str(record.get(key) or "")
            for key in ["run_id", "model_id", "experiment_path"]
        ).lower()
        if "baseline" in searchable:
            return record
    for record in ordered:
        model_name = str(record.get("model_name") or record.get("model_id") or "")
        if model_name.startswith("logistic_regression"):
            return record
    return ordered[0] if ordered else None


def metric_delta(record: dict[str, Any], baseline: dict[str, Any] | None, primary_metric: str) -> float | None:
    value = validation_metric_value(record, primary_metric)
    baseline_value = validation_metric_value(baseline or {}, primary_metric)
    if value is None or baseline_value is None:
        return None
    if metric_lower_is_better(primary_metric):
        return baseline_value - value
    return value - baseline_value


def test_metric_value(record: dict[str, Any], metric_name: str) -> float | None:
    value = record.get("test_metrics", {}).get(metric_name)
    if isinstance(value, (int, float)) and pd.notna(value):
        return float(value)
    return None


def validation_test_gap(record: dict[str, Any], primary_metric: str) -> float | None:
    validation_value = validation_metric_value(record, primary_metric)
    test_value = test_metric_value(record, primary_metric)
    if validation_value is None or test_value is None:
        return None
    return abs(validation_value - test_value)


def default_selected_index(registry: list[dict[str, Any]]) -> int:
    latest = registry[-1] if registry else None
    if latest is None:
        return 0
    primary_metric = str(latest.get("primary_metric") or "macro_f1")
    best = best_model_record(registry, latest.get("data_version"), primary_metric)
    if best is None:
        return len(registry) - 1
    for index, record in enumerate(registry):
        if same_record(record, best):
            return index
    return len(registry) - 1


def resolve_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else BASE_DIR / path


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_yaml(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_registry() -> list[dict[str, Any]]:
    registry = read_json(REGISTRY_PATH, [])
    return registry if isinstance(registry, list) else []


@st.cache_data
def load_metrics(experiment_path: str) -> dict[str, Any]:
    experiment_dir = resolve_path(experiment_path)
    if experiment_dir is None:
        return {}
    return read_json(experiment_dir / "metrics.json", {})


@st.cache_data
def load_run_config(experiment_path: str) -> dict[str, Any]:
    experiment_dir = resolve_path(experiment_path)
    if experiment_dir is None:
        return {}
    config = read_yaml(experiment_dir / "config.yaml", {})
    return config if isinstance(config, dict) else {}


@st.cache_data
def load_confusion_matrix(path: str) -> dict[str, Any]:
    resolved = resolve_path(path)
    if resolved is None:
        return {}
    return read_json(resolved, {})


@st.cache_data
def load_csv_artifact(path: str | None) -> pd.DataFrame:
    resolved = resolve_path(path)
    if resolved is None or not resolved.exists():
        return pd.DataFrame()
    return pd.read_csv(resolved)


@st.cache_data
def load_automl_summary(path: str | None) -> dict[str, Any]:
    resolved = resolve_path(path)
    if resolved is None or not resolved.exists():
        return {}
    summary = read_json(resolved, {})
    return summary if isinstance(summary, dict) else {}


def selected_automl_model(record: dict[str, Any], summary: dict[str, Any]) -> str:
    return str(summary.get("selected_model") or record.get("model_name") or "n/a")


def automl_model_options(predictor: Any, leaderboard: pd.DataFrame, selected_model: str) -> list[str]:
    names = autogluon_model_names(predictor, leaderboard)
    options = [name for name in [selected_model, *names] if name and name != "n/a"]
    return list(dict.fromkeys(options))


def metric_direction_label(metric_name: str) -> str:
    return "lower is better" if metric_lower_is_better(metric_name) else "higher is better"


def metric_delta_value(value: Any, reference: Any, metric_name: str) -> float | None:
    value_float = scalar_float(value)
    reference_float = scalar_float(reference)
    if value_float is None or reference_float is None:
        return None
    return reference_float - value_float if metric_lower_is_better(metric_name) else value_float - reference_float


def format_seconds(value: Any) -> str:
    seconds = scalar_float(value)
    if seconds is None:
        return "n/a"
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.3f} s"


def format_multiplier(value: Any) -> str:
    multiplier = scalar_float(value)
    if multiplier is None:
        return "n/a"
    if multiplier == 0:
        return "0x"
    return f"{multiplier:.1f}x"


def format_optional_delta(value: Any) -> str:
    delta = scalar_float(value)
    return format_delta(delta) if delta is not None else "n/a"


def automl_metric_rows(record: dict[str, Any], baseline: dict[str, Any] | None, primary_metric: str) -> pd.DataFrame:
    rows = [
        {
            "item": f"validation {primary_metric}",
            "value": format_metric(validation_metric_value(record, primary_metric)),
            "meaning": "model selection 기준",
        },
        {
            "item": f"test {primary_metric}",
            "value": format_metric(test_metric_value(record, primary_metric)),
            "meaning": "최종 확인용",
        },
        {
            "item": "delta vs baseline",
            "value": format_delta(metric_delta(record, baseline, primary_metric)),
            "meaning": "같은 data_version 기준",
        },
        {
            "item": "validation-test gap",
            "value": format_metric(validation_test_gap(record, primary_metric)),
            "meaning": "작을수록 안정적",
        },
    ]
    return pd.DataFrame(rows)


def numeric_row_value(row: pd.Series, column: str) -> float | None:
    if column not in row:
        return None
    return scalar_float(row.get(column))


def is_ensemble_model_name(model_name: Any) -> bool:
    return "ensemble" in str(model_name or "").lower()


def is_single_model_row(row: pd.Series) -> bool:
    stack_level = numeric_row_value(row, "stack_level")
    if stack_level is not None and stack_level > 1:
        return False
    return not is_ensemble_model_name(row.get("model"))


def best_single_model_row(leaderboard: pd.DataFrame) -> pd.Series | None:
    if leaderboard.empty or "model" not in leaderboard.columns:
        return None
    for _, row in leaderboard.iterrows():
        if is_single_model_row(row):
            return row
    return None


def prediction_time_value(row: pd.Series | None) -> float | None:
    if row is None:
        return None
    test_time = numeric_row_value(row, "pred_time_test")
    if test_time is not None:
        return test_time
    return numeric_row_value(row, "pred_time_val")


def automl_candidate_dataframe(
    leaderboard: pd.DataFrame,
    selected_model: str,
    primary_metric: str,
) -> pd.DataFrame:
    if leaderboard.empty:
        return pd.DataFrame()
    best_single = best_single_model_row(leaderboard)
    best_single_name = str(best_single.get("model")) if best_single is not None else None
    best_single_val = numeric_row_value(best_single, "score_val") if best_single is not None else None
    best_single_test = numeric_row_value(best_single, "score_test") if best_single is not None else None
    selected_row = selected_leaderboard_row(leaderboard, selected_model)
    selected_time = prediction_time_value(selected_row)
    rows = []
    for _, row in leaderboard.iterrows():
        raw_model = str(row.get("model") or "n/a")
        validation_score = numeric_row_value(row, "score_val")
        test_score = numeric_row_value(row, "score_test")
        prediction_time = prediction_time_value(row)
        stack_level = numeric_row_value(row, "stack_level")
        if raw_model == selected_model:
            role = "Selected"
        elif best_single_name and raw_model == best_single_name:
            role = "Best single model"
        else:
            role = "Candidate"
        rows.append(
            {
                "raw_model": raw_model,
                "role": role,
                "model": display_model_name(None, raw_model),
                "is_ensemble": is_ensemble_model_name(raw_model) or (stack_level is not None and stack_level > 1),
                "validation_score": validation_score,
                "test_score": test_score,
                "validation_test_gap": abs(validation_score - test_score)
                if validation_score is not None and test_score is not None
                else None,
                "prediction_time_s": prediction_time,
                "fit_time_s": numeric_row_value(row, "fit_time"),
                "speed_vs_selected": prediction_time / selected_time
                if prediction_time is not None and selected_time and selected_time > 0
                else None,
                "validation_delta_vs_best_single": metric_delta_value(
                    validation_score,
                    best_single_val,
                    primary_metric,
                ),
                "test_delta_vs_best_single": metric_delta_value(test_score, best_single_test, primary_metric),
                "stack_level": int(stack_level) if stack_level is not None else None,
                "can_infer": row.get("can_infer"),
            }
        )
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return candidates

    rank_ascending = metric_lower_is_better(primary_metric)
    for score_column, rank_column in [
        ("validation_score", "validation_rank"),
        ("test_score", "test_rank"),
    ]:
        if score_column in candidates:
            candidates[rank_column] = (
                pd.to_numeric(candidates[score_column], errors="coerce")
                .rank(method="min", ascending=rank_ascending)
                .astype("Int64")
            )
    return candidates


def automl_display_leaderboard(
    leaderboard: pd.DataFrame,
    selected_model: str,
    primary_metric: str,
) -> pd.DataFrame:
    candidates = automl_candidate_dataframe(leaderboard, selected_model, primary_metric)
    return candidate_display_dataframe(candidates)


def candidate_display_dataframe(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    display_columns = [
        "role",
        "model",
        "validation_rank",
        "test_rank",
        "validation_score",
        "test_score",
        "validation_test_gap",
        "prediction_time_s",
        "speed_vs_selected",
        "validation_delta_vs_best_single",
        "test_delta_vs_best_single",
        "fit_time_s",
    ]
    display_df = candidates[[column for column in display_columns if column in candidates.columns]].copy()
    for column in [
        "validation_score",
        "test_score",
        "validation_test_gap",
        "prediction_time_s",
        "speed_vs_selected",
        "validation_delta_vs_best_single",
        "test_delta_vs_best_single",
        "fit_time_s",
    ]:
        if column in display_df.columns:
            display_df[column] = pd.to_numeric(display_df[column], errors="coerce").round(4)
    return display_df


def selected_leaderboard_row(
    leaderboard: pd.DataFrame,
    selected_model: str,
) -> pd.Series | None:
    if leaderboard.empty or "model" not in leaderboard.columns:
        return None
    matches = leaderboard[leaderboard["model"].astype(str) == selected_model]
    if not matches.empty:
        return matches.iloc[0]
    return leaderboard.iloc[0]


def artifact_status_dataframe(record: dict[str, Any]) -> pd.DataFrame:
    artifact_fields = [
        ("Model artifact", "artifact_path"),
        ("Run directory", "experiment_path"),
        ("Leaderboard", "leaderboard_path"),
        ("AutoML summary", "automl_summary_path"),
        ("Confusion matrix", "confusion_matrix_path"),
        ("Threshold metrics", "threshold_metrics_path"),
    ]
    rows = []
    for label, field in artifact_fields:
        path_value = record.get(field)
        if not path_value:
            continue
        resolved = resolve_path(path_value)
        rows.append(
            {
                "artifact": label,
                "configured_path": str(path_value),
                "resolved_path": str(resolved) if resolved else "n/a",
                "status": "found" if resolved and resolved.exists() else "missing",
            }
        )
    return pd.DataFrame(rows)


def missing_artifact_labels(record: dict[str, Any], fields: list[tuple[str, str]]) -> list[str]:
    missing = []
    for label, field in fields:
        path_value = record.get(field)
        resolved = resolve_path(path_value)
        if path_value and (resolved is None or not resolved.exists()):
            missing.append(f"{label}: {path_value}")
    return missing


def render_run_source_diagnostics(record: dict[str, Any]) -> None:
    with st.expander("Run Source"):
        st.write(f"BASE_DIR: `{BASE_DIR}`")
        st.write(f"registry: `{REGISTRY_PATH}`")
        st.write(f"registry_exists: `{REGISTRY_PATH.exists()}`")
        status_df = artifact_status_dataframe(record)
        if status_df.empty:
            st.info("선택한 run에 artifact 경로가 기록되어 있지 않습니다.")
        else:
            st.dataframe(status_df, use_container_width=True, hide_index=True)


def metric_comparison_dataframe(record: dict[str, Any]) -> pd.DataFrame:
    validation_metrics = record.get("metrics", {})
    test_metrics = record.get("test_metrics", {})
    metric_names = sorted(
        {
            key
            for source in [validation_metrics, test_metrics]
            for key, value in source.items()
            if isinstance(value, (int, float))
        }
    )
    rows = []
    for metric_name in metric_names:
        validation_value = scalar_float(validation_metrics.get(metric_name))
        test_value = scalar_float(test_metrics.get(metric_name))
        rows.append(
            {
                "metric": metric_name,
                "validation": validation_value,
                "test": test_value,
                "gap": abs(validation_value - test_value)
                if validation_value is not None and test_value is not None
                else None,
            }
        )
    return pd.DataFrame(rows)


def metric_comparison_long(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()
    long_df = metrics_df.melt(
        id_vars=["metric", "gap"],
        value_vars=[column for column in ["validation", "test"] if column in metrics_df.columns],
        var_name="split",
        value_name="value",
    )
    return long_df.dropna(subset=["value"])


def threshold_curve_dataframe(threshold_df: pd.DataFrame) -> pd.DataFrame:
    if threshold_df.empty:
        return pd.DataFrame()
    numeric_df = threshold_df.copy()
    for column in ["threshold", "precision", "recall", "f1", "tp", "fp", "tn", "fn"]:
        if column in numeric_df.columns:
            numeric_df[column] = pd.to_numeric(numeric_df[column], errors="coerce")
    value_columns = [column for column in ["precision", "recall", "f1"] if column in numeric_df.columns]
    if "threshold" not in numeric_df.columns or not value_columns:
        return pd.DataFrame()
    return numeric_df.melt(
        id_vars=["threshold"],
        value_vars=value_columns,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["threshold", "value"])


def confusion_matrix_long_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    labels = [str(label) for label in payload.get("labels", [])]
    matrix = payload.get("matrix", [])
    rows = []
    for row_index, actual_label in enumerate(labels):
        row_values = matrix[row_index] if row_index < len(matrix) else []
        row_total = sum(row_values) if row_values else 0
        for column_index, predicted_label in enumerate(labels):
            count = row_values[column_index] if column_index < len(row_values) else 0
            rows.append(
                {
                    "actual": f"actual {actual_label}",
                    "predicted": f"pred {predicted_label}",
                    "count": int(count),
                    "row_percent": count / row_total if row_total else 0.0,
                }
            )
    return pd.DataFrame(rows)


def candidate_chart_df(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    chart_df = candidates.copy()
    for column in [
        "validation_score",
        "test_score",
        "validation_test_gap",
        "prediction_time_s",
        "fit_time_s",
    ]:
        if column in chart_df.columns:
            chart_df[column] = pd.to_numeric(chart_df[column], errors="coerce")
    return chart_df


def role_color_scale() -> alt.Scale:
    return alt.Scale(
        domain=["Selected", "Best single model", "Candidate"],
        range=["#176B87", "#D97706", "#64748B"],
    )


@st.cache_resource
def load_model(artifact_path: str) -> Any:
    resolved = resolve_path(artifact_path)
    if resolved is None:
        raise FileNotFoundError("artifact_path가 비어 있습니다.")
    with open(resolved, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_autogluon_model(artifact_path: str) -> Any:
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise ImportError(
            "AutoGluon이 설치되어 있지 않습니다. `pip install -r requirements.txt` 후 다시 실행하세요."
        ) from exc

    resolved = resolve_path(artifact_path)
    if resolved is None:
        raise FileNotFoundError("artifact_path가 비어 있습니다.")
    return TabularPredictor.load(str(resolved))


@st.cache_data
def load_training_data(data_path: str | None) -> pd.DataFrame:
    resolved = resolve_path(data_path)
    if resolved is None or not resolved.exists():
        return pd.DataFrame()
    return pd.read_csv(resolved)


@st.cache_data
def load_logs() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        return pd.DataFrame()

    logs = pd.DataFrame(records)
    if "timestamp" in logs.columns:
        logs["timestamp"] = pd.to_datetime(logs["timestamp"], errors="coerce")
        logs = logs.sort_values("timestamp", ascending=False)
    if "latency_ms" in logs.columns:
        logs["latency_ms"] = pd.to_numeric(logs["latency_ms"], errors="coerce")
    if "probability" in logs.columns:
        logs["probability"] = pd.to_numeric(logs["probability"], errors="coerce")
    return logs


def registry_to_dataframe(
    registry: list[dict[str, Any]],
    primary_metric: str,
    best_record: dict[str, Any] | None,
    baseline: dict[str, Any] | None,
) -> pd.DataFrame:
    rows = []
    ranked = ranked_records(registry, primary_metric)
    rank_lookup = {record_key(record): index + 1 for index, record in enumerate(ranked)}
    for record in ranked:
        metrics = record.get("metrics", {})
        test_metrics = record.get("test_metrics", {})
        rank = rank_lookup.get(record_key(record))
        delta = metric_delta(record, baseline, primary_metric)
        rows.append(
            {
                "rank": rank if validation_metric_value(record, primary_metric) is not None else "n/a",
                "is_best": "yes" if same_record(record, best_record) else "",
                "delta_vs_baseline": format_delta(delta),
                "created_at": record.get("created_at"),
                "model": display_model_name(record),
                "experiment": display_experiment_type(record),
                "data": display_data_name(record.get("data_version")),
                "primary_metric": record.get("primary_metric"),
                "selection_metric": format_metric(metrics.get(primary_metric)),
                "val_accuracy": format_metric(metrics.get("accuracy")),
                "val_macro_f1": format_metric(metrics.get("macro_f1")),
                "test_accuracy": format_metric(test_metrics.get("accuracy")),
                "test_macro_f1": format_metric(test_metrics.get("macro_f1")),
            }
        )
    return pd.DataFrame(rows)


def registry_trend_dataframe(registry: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in registry:
        metrics = record.get("metrics", {})
        rows.append(
            {
                "created_at": record.get("created_at"),
                "val_accuracy": metrics.get("accuracy"),
                "val_macro_f1": metrics.get("macro_f1"),
            }
        )
    return pd.DataFrame(rows)


def model_label(record: dict[str, Any]) -> str:
    primary_metric = record.get("primary_metric", "macro_f1")
    metric = record.get("metrics", {}).get(primary_metric)
    metric_text = format_metric(metric)
    return f"{display_run_label(record)} · val {primary_metric}={metric_text}"


def class_report_dataframe(metrics: dict[str, Any]) -> pd.DataFrame:
    report = metrics.get("classification_report", {})
    rows = []
    for label, values in report.items():
        if not isinstance(values, dict) or not {"precision", "recall", "f1-score"} <= set(values):
            continue
        rows.append(
            {
                "class": str(label),
                "precision": values.get("precision"),
                "recall": values.get("recall"),
                "f1": values.get("f1-score"),
                "support": values.get("support"),
            }
        )
    return pd.DataFrame(rows)


def confusion_matrix_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    labels = [str(label) for label in payload.get("labels", [])]
    matrix = payload.get("matrix", [])
    if not labels or not matrix:
        return pd.DataFrame()
    index = [f"actual {label}" for label in labels]
    columns = [f"pred {label}" for label in labels]
    return pd.DataFrame(matrix, index=index, columns=columns)


def run_data_config(config: dict[str, Any]) -> dict[str, Any]:
    data_config = config.get("data", {})
    return data_config if isinstance(data_config, dict) else {}


def target_column(config: dict[str, Any]) -> str | None:
    target = run_data_config(config).get("target_column")
    return str(target) if target else None


def model_feature_columns(model: Any, config: dict[str, Any], data: pd.DataFrame) -> list[str]:
    try:
        feature_metadata = getattr(model, "feature_metadata", None)
        if feature_metadata is not None:
            features = feature_metadata.get_features()
            if features:
                return [str(column) for column in features]
    except (AttributeError, TypeError, ValueError):
        pass

    try:
        preprocessor = model.named_steps["preprocessor"]
        columns: list[str] = []
        for _, _, transformer_columns in preprocessor.transformers_:
            if isinstance(transformer_columns, (list, tuple, pd.Index)):
                columns.extend(str(column) for column in transformer_columns)
        if columns:
            return columns
    except (AttributeError, KeyError, ValueError):
        pass

    try:
        return [str(column) for column in model.named_steps["scaler"].feature_names_in_]
    except (AttributeError, KeyError):
        pass

    target = target_column(config)
    if not data.empty and target in data.columns:
        return [str(column) for column in data.columns if column != target]
    return []


def default_numeric_value(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    value = numeric.median()
    if pd.isna(value):
        return 0.0
    return float(value)


def is_integer_numeric(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return False
    return bool((numeric.round() == numeric).all())


def feature_kind(column: str, data: pd.DataFrame) -> str:
    if column not in data.columns:
        return "unknown"
    series = data[column]
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "integer" if is_integer_numeric(series) else "float"
    return "categorical"


def feature_counts(feature_columns: list[str], data: pd.DataFrame) -> tuple[int, int, int]:
    numeric_count = 0
    categorical_count = 0
    for column in feature_columns:
        kind = feature_kind(column, data)
        if kind in {"integer", "float"}:
            numeric_count += 1
        else:
            categorical_count += 1
    return len(feature_columns), numeric_count, categorical_count


def feature_details_dataframe(feature_columns: list[str], data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in feature_columns:
        kind = feature_kind(column, data)
        if column in data.columns:
            series = data[column]
            missing_rate = float(series.isna().mean())
            if kind in {"integer", "float"}:
                default_value = default_numeric_value(series)
                default_text = f"{default_value:.0f}" if kind == "integer" else f"{default_value:.3f}"
            else:
                options = categorical_options(series)
                default_text = str(options[0]) if options else ""
        else:
            missing_rate = 0.0
            default_text = ""
        rows.append(
            {
                "feature": column,
                "type": kind,
                "missing_rate": format_percent(missing_rate),
                "default": default_text,
            }
        )
    return pd.DataFrame(rows)


def categorical_options(series: pd.Series) -> list[Any]:
    values = series.dropna()
    ordered: list[Any] = []
    mode = values.mode()
    if not mode.empty:
        ordered.extend(mode.tolist())
    for value in values.drop_duplicates().tolist():
        if value not in ordered:
            ordered.append(value)
    if series.isna().any():
        ordered.append(MISSING_OPTION)
    return ordered[:50]


def render_feature_input(column: str, data: pd.DataFrame) -> Any:
    if column not in data.columns:
        return st.text_input(column, value="")

    series = data[column]
    if pd.api.types.is_bool_dtype(series):
        default = bool(series.dropna().mode().iloc[0]) if not series.dropna().empty else False
        return st.checkbox(column, value=default)
    if pd.api.types.is_numeric_dtype(series):
        default = default_numeric_value(series)
        if is_integer_numeric(series):
            return st.number_input(column, value=int(round(default)), step=1)
        return st.number_input(column, value=default, step=0.1, format="%.3f")

    options = categorical_options(series)
    if options:
        value = st.selectbox(column, options=options)
        return None if value == MISSING_OPTION else value
    return st.text_input(column, value="")


def prediction_probabilities(
    model: Any,
    features: pd.DataFrame,
    model_name: str | None = None,
) -> dict[str, float]:
    if not hasattr(model, "predict_proba"):
        return {}
    if model_name:
        try:
            probabilities = model.predict_proba(features, model=model_name, as_multiclass=True)
        except TypeError:
            probabilities = model.predict_proba(features, model=model_name)
    else:
        probabilities = model.predict_proba(features)

    if isinstance(probabilities, pd.DataFrame):
        first_row = probabilities.iloc[0]
        return {str(label): float(value) for label, value in first_row.items()}

    if isinstance(probabilities, pd.Series):
        positive_class = getattr(model, "positive_class", None)
        class_labels = [str(value) for value in getattr(model, "class_labels", [])]
        if positive_class is None and len(class_labels) == 2:
            positive_class = class_labels[1]
        positive_label = str(positive_class or "positive")
        negative_label = class_labels[0] if len(class_labels) == 2 else "negative"
        positive_probability = float(probabilities.iloc[0])
        return {
            str(negative_label): 1.0 - positive_probability,
            positive_label: positive_probability,
        }

    probabilities_array = probabilities[0]
    classes = [str(value) for value in getattr(model, "classes_", [])]
    return {class_label: float(probabilities_array[index]) for index, class_label in enumerate(classes)}


def prediction_value(model: Any, features: pd.DataFrame, model_name: str | None = None) -> Any:
    if model_name:
        try:
            return first_prediction_value(model.predict(features, model=model_name))
        except TypeError:
            return first_prediction_value(model.predict(features))
    return first_prediction_value(model.predict(features))


def first_prediction_value(prediction: Any) -> Any:
    if hasattr(prediction, "iloc"):
        return prediction.iloc[0]
    array = np.asarray(prediction)
    return array[0] if array.size else None


def autogluon_model_names(predictor: Any, leaderboard: pd.DataFrame) -> list[str]:
    if not leaderboard.empty and "model" in leaderboard.columns:
        names = [str(value) for value in leaderboard["model"].dropna().tolist()]
        if names:
            return list(dict.fromkeys(names))

    for method_name in ["model_names", "get_model_names"]:
        method = getattr(predictor, method_name, None)
        if callable(method):
            try:
                names = [str(value) for value in method()]
            except TypeError:
                names = [str(value) for value in method(can_infer=True)]
            if names:
                return list(dict.fromkeys(names))
    return []


def default_explanation_models(model_names: list[str], leaderboard: pd.DataFrame) -> list[str]:
    if not leaderboard.empty and "model" in leaderboard.columns:
        singles = [str(row.get("model")) for _, row in leaderboard.iterrows() if is_single_model_row(row)]
    else:
        singles = [name for name in model_names if not is_ensemble_model_name(name)]
    defaults = [name for name in singles if name in model_names][:2]
    return defaults if len(defaults) == 2 else model_names[:2]


def autogluon_probability_dict(predictor: Any, features: pd.DataFrame, model_name: str) -> dict[str, float]:
    try:
        probabilities = predictor.predict_proba(features, model=model_name, as_multiclass=True)
    except TypeError:
        probabilities = predictor.predict_proba(features, model=model_name)

    if isinstance(probabilities, pd.DataFrame):
        first_row = probabilities.iloc[0]
        return {str(label): float(value) for label, value in first_row.items()}
    if isinstance(probabilities, pd.Series):
        positive_class = str(getattr(predictor, "positive_class", "") or "positive")
        labels = [str(value) for value in getattr(predictor, "class_labels", [])]
        negative_class = next((label for label in labels if label != positive_class), "negative")
        positive_probability = float(probabilities.iloc[0])
        return {
            negative_class: 1.0 - positive_probability,
            positive_class: positive_probability,
        }
    array = np.asarray(probabilities)
    if array.ndim == 1:
        positive_class = str(getattr(predictor, "positive_class", "") or "positive")
        negative_class = "negative"
        return {negative_class: float(1.0 - array[0]), positive_class: float(array[0])}
    labels = [str(value) for value in getattr(predictor, "class_labels", [])]
    return {label: float(array[0, index]) for index, label in enumerate(labels)}


def autogluon_prediction_row(
    predictor: Any,
    features: pd.DataFrame,
    model_name: str,
    task_type: str,
    positive_class: str | None,
) -> tuple[dict[str, Any], dict[str, float]]:
    prediction = first_prediction_value(predictor.predict(features, model=model_name))
    probabilities = autogluon_probability_dict(predictor, features, model_name) if task_type == "classification" else {}
    target_output = None
    if task_type == "classification":
        if positive_class and positive_class in probabilities and len(probabilities) == 2:
            target_output = positive_class
        else:
            target_output = str(prediction)
    confidence = probabilities.get(str(prediction)) if probabilities else None
    if task_type == "classification" and target_output in probabilities:
        target_value = probabilities[target_output]
    else:
        target_value = float(prediction) if isinstance(prediction, (int, float, np.integer, np.floating)) else None
    return {
        "model": model_name,
        "prediction": str(prediction),
        "confidence": format_percent(confidence) if probabilities else "n/a",
        "explanation_target": target_output if target_output is not None else "prediction",
        "target_value": format_metric(target_value),
    }, probabilities


def coerce_like_reference(data: Any, columns: list[str], reference: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(data, columns=columns)
    for column in columns:
        if column in reference.columns and pd.api.types.is_numeric_dtype(reference[column]):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def scalar_float(value: Any) -> float | None:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.size == 0:
        return None
    result = float(array.flat[0])
    return result if pd.notna(result) else None


def shap_background_data(training_data: pd.DataFrame, feature_columns: list[str], target: str | None) -> pd.DataFrame:
    if training_data.empty:
        return pd.DataFrame()
    candidates = training_data.drop(columns=[target], errors="ignore")
    missing_columns = [column for column in feature_columns if column not in candidates.columns]
    if missing_columns:
        return pd.DataFrame()
    background = candidates[feature_columns].dropna(how="all")
    if background.empty:
        return pd.DataFrame()
    sample_size = min(50, len(background))
    return background.sample(n=sample_size, random_state=42) if len(background) > sample_size else background


def shap_values_for_model(
    predictor: Any,
    model_name: str,
    features: pd.DataFrame,
    background: pd.DataFrame,
    task_type: str,
    target_output: str | None,
) -> dict[str, Any]:
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP이 설치되어 있지 않습니다. `pip install -r requirements.txt` 후 다시 실행하세요."
        ) from exc

    feature_columns = features.columns.tolist()

    def model_output(batch: Any) -> np.ndarray:
        batch_df = coerce_like_reference(batch, feature_columns, background)
        if task_type == "regression":
            return np.asarray(predictor.predict(batch_df, model=model_name), dtype=float)

        probabilities = predictor.predict_proba(batch_df, model=model_name, as_multiclass=True)
        if isinstance(probabilities, pd.Series):
            return probabilities.to_numpy(dtype=float)
        if isinstance(probabilities, pd.DataFrame):
            if target_output in probabilities.columns:
                return probabilities[target_output].to_numpy(dtype=float)
            string_lookup = {str(column): column for column in probabilities.columns}
            if target_output in string_lookup:
                return probabilities[string_lookup[target_output]].to_numpy(dtype=float)
            return probabilities.iloc[:, 0].to_numpy(dtype=float)
        array = np.asarray(probabilities, dtype=float)
        if array.ndim == 1:
            return array
        return array[:, 0]

    explainer = shap.KernelExplainer(model_output, background)
    try:
        raw_values = explainer.shap_values(features, nsamples=100, silent=True)
    except TypeError:
        raw_values = explainer.shap_values(features, nsamples=100)
    if isinstance(raw_values, list):
        raw_values = raw_values[0]
    values = np.asarray(raw_values)
    while values.ndim > 1:
        values = values[0]
    shap_values = values.astype(float)
    base_value = scalar_float(getattr(explainer, "expected_value", None))
    model_output_value = scalar_float(model_output(features))
    if model_output_value is None and base_value is not None:
        model_output_value = base_value + float(np.sum(shap_values))
    return {
        "values": shap_values,
        "base_value": base_value,
        "model_output": model_output_value,
    }


def shap_table(feature_columns: list[str], values: np.ndarray, sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, shap_value in zip(feature_columns, values):
        rows.append(
            {
                "feature": feature,
                "input_value": sample.iloc[0][feature],
                "shap_value": float(shap_value),
                "direction": "increases target output" if shap_value >= 0 else "decreases target output",
                "abs_shap": abs(float(shap_value)),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_shap", ascending=False)


def shap_summary_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    values = np.asarray(payload.get("values", []), dtype=float)
    base_value = payload.get("base_value")
    model_output_value = payload.get("model_output")
    shap_sum = float(np.sum(values)) if values.size else None
    reconstructed = base_value + shap_sum if isinstance(base_value, (int, float)) and shap_sum is not None else None
    rows = [
        {"item": "Base output", "value": format_metric(base_value), "meaning": "background sample 평균 출력"},
        {"item": "SHAP sum", "value": format_delta(shap_sum), "meaning": "이 row의 feature 기여도 합"},
        {"item": "Final output", "value": format_metric(model_output_value), "meaning": "설명 target의 모델 출력"},
    ]
    if reconstructed is not None:
        rows.append({"item": "Base + SHAP sum", "value": format_metric(reconstructed), "meaning": "SHAP additive check"})
    return pd.DataFrame(rows)


def format_feature_value(value: Any) -> str:
    if isinstance(value, (int, float, np.integer, np.floating)) and pd.notna(value):
        numeric = float(value)
        return f"{numeric:,.0f}" if numeric.is_integer() else f"{numeric:,.3f}"
    return str(value)


def shap_diverging_chart_html(explanation: pd.DataFrame) -> str:
    chart_df = explanation.copy()
    chart_df["impact"] = np.where(chart_df["shap_value"] >= 0, "increases target", "decreases target")
    max_abs = float(chart_df["abs_shap"].max()) if not chart_df.empty else 1.0
    max_abs = max(max_abs, 1e-12)
    rows = []
    for _, row in chart_df.iterrows():
        shap_value = float(row["shap_value"])
        width = min(abs(shap_value) / max_abs * 48.0, 48.0)
        if shap_value >= 0:
            bar_style = f"left:50%;width:{width:.2f}%;background:#d94f45;"
            value_class = "positive"
        else:
            bar_style = f"left:{50.0 - width:.2f}%;width:{width:.2f}%;background:#2f6bc2;"
            value_class = "negative"
        rows.append(
            textwrap.dedent(f"""
            <div class="shap-row">
              <div class="shap-feature">
                <strong>{html.escape(str(row["feature"]))}</strong>
                <span>value={html.escape(format_feature_value(row["input_value"]))}</span>
              </div>
              <div class="shap-track">
                <div class="shap-zero"></div>
                <div class="shap-bar" style="{bar_style}"></div>
              </div>
              <div class="shap-value {value_class}">{shap_value:+.4f}</div>
            </div>
            """).strip()
        )
    legend = textwrap.dedent("""
    <div class="shap-legend">
      <span><i class="neg"></i>decreases target</span>
      <span><i class="pos"></i>increases target</span>
    </div>
    """).strip()
    return textwrap.dedent(f"""
    <style>
      .shap-legend {{
        display: flex;
        gap: 1rem;
        margin: 0.25rem 0 0.75rem 0;
        color: #6b7280;
        font-size: 0.86rem;
      }}
      .shap-legend i {{
        display: inline-block;
        width: 0.75rem;
        height: 0.75rem;
        border-radius: 0.15rem;
        margin-right: 0.35rem;
        vertical-align: -0.08rem;
      }}
      .shap-legend .neg {{ background: #2f6bc2; }}
      .shap-legend .pos {{ background: #d94f45; }}
      .shap-row {{
        display: grid;
        grid-template-columns: minmax(6.5rem, 9rem) minmax(12rem, 1fr) 5.25rem;
        align-items: center;
        gap: 0.7rem;
        margin: 0.5rem 0;
      }}
      .shap-feature {{
        display: flex;
        flex-direction: column;
        min-width: 0;
        line-height: 1.12;
      }}
      .shap-feature strong {{
        color: #1f2937;
        font-size: 0.9rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }}
      .shap-feature span {{
        color: #6b7280;
        font-size: 0.72rem;
        margin-top: 0.18rem;
      }}
      .shap-track {{
        position: relative;
        height: 1.35rem;
        background: #f3f4f6;
        border-radius: 0.25rem;
        overflow: hidden;
      }}
      .shap-zero {{
        position: absolute;
        left: 50%;
        top: 0;
        width: 1px;
        height: 100%;
        background: #6b7280;
        opacity: 0.75;
      }}
      .shap-bar {{
        position: absolute;
        top: 0.25rem;
        height: 0.85rem;
        border-radius: 0.18rem;
      }}
      .shap-value {{
        text-align: right;
        font-variant-numeric: tabular-nums;
        font-size: 0.86rem;
      }}
      .shap-value.negative {{ color: #2f6bc2; }}
      .shap-value.positive {{ color: #b83b35; }}
    </style>
    {legend}
    {''.join(rows)}
    """).strip()


def shap_comparison_dataframe(model_payloads: list[dict[str, Any]]) -> pd.DataFrame:
    if len(model_payloads) != 2:
        return pd.DataFrame()
    left, right = model_payloads
    left_df = left["explanation"][["feature", "shap_value"]].rename(
        columns={"shap_value": f"{left['display_name']} shap"}
    )
    right_df = right["explanation"][["feature", "shap_value"]].rename(
        columns={"shap_value": f"{right['display_name']} shap"}
    )
    merged = left_df.merge(right_df, on="feature", how="outer").fillna(0.0)
    left_col = f"{left['display_name']} shap"
    right_col = f"{right['display_name']} shap"
    merged["difference"] = (merged[left_col] - merged[right_col]).abs()
    merged["same_direction"] = np.sign(merged[left_col]) == np.sign(merged[right_col])
    return merged.sort_values("difference", ascending=False).head(10)


def render_autogluon_local_explanation(
    record: dict[str, Any],
    predictor: Any,
    config: dict[str, Any],
    training_data: pd.DataFrame,
    feature_columns: list[str],
    features: pd.DataFrame,
) -> None:
    if not is_automl_record(record):
        return

    st.divider()
    st.subheader("Local Explanation")
    st.caption(
        "입력한 1개 row를 AutoGluon 내부 모델 2개에 넣고 SHAP으로 비교합니다. "
        "이 값은 모델 판단의 근거 후보이며 인과 설명은 아닙니다."
    )

    leaderboard = load_csv_artifact(record.get("leaderboard_path"))
    model_names = autogluon_model_names(predictor, leaderboard)
    if len(model_names) < 2:
        st.info("AutoGluon 내부 모델이 2개 이상 있을 때 local explanation을 비교할 수 있습니다.")
        return

    default_models = default_explanation_models(model_names, leaderboard)
    selected_models = st.multiselect(
        "Compare AutoGluon models",
        options=model_names,
        default=default_models,
        format_func=lambda name: display_model_name(record, name),
        key=f"local_explanation_models_{record.get('run_id')}",
    )
    if len(selected_models) != 2:
        st.info("비교할 AutoGluon 모델을 정확히 2개 선택하세요.")
        return

    target = target_column(config)
    background = shap_background_data(training_data, feature_columns, target)
    if background.empty:
        st.info("SHAP background data를 만들 수 없습니다. run config의 processed CSV와 feature columns를 확인하세요.")
        return

    task_type = str(record.get("task_type") or "classification")
    positive_class = str(record.get("positive_class") or "") or None

    if not st.button("Run local explanation", use_container_width=True):
        return

    comparison_rows = []
    model_payloads = []
    for model_name in selected_models:
        try:
            prediction_row, probabilities = autogluon_prediction_row(
                predictor,
                features,
                model_name,
                task_type,
                positive_class,
            )
            prediction_row["model"] = display_model_name(record, model_name)
            prediction_row["raw_model"] = model_name
            comparison_rows.append(prediction_row)
            model_payloads.append((model_name, prediction_row, probabilities))
        except Exception as exc:  # noqa: BLE001
            comparison_rows.append({
                "model": display_model_name(record, model_name),
                "raw_model": model_name,
                "prediction": "error",
                "confidence": "n/a",
                "explanation_target": "n/a",
                "target_value": "n/a",
                "error": str(exc),
            })

    st.markdown("#### Model Responses")
    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

    explanation_payloads = []
    columns = st.columns(2)
    for column, (model_name, prediction_row, _) in zip(columns, model_payloads):
        with column:
            display_name = display_model_name(record, model_name)
            st.markdown(f"#### {display_name}")
            st.caption(f"raw_model={model_name}")
            try:
                with st.spinner("Computing SHAP values..."):
                    shap_payload = shap_values_for_model(
                        predictor=predictor,
                        model_name=model_name,
                        features=features,
                        background=background,
                        task_type=task_type,
                        target_output=prediction_row.get("explanation_target"),
                    )
                values = np.asarray(shap_payload.get("values", []), dtype=float)
                explanation_full = shap_table(feature_columns, values, features)
                explanation = explanation_full.head(10)
                explanation_payloads.append(
                    {
                        "display_name": display_name,
                        "raw_model": model_name,
                        "explanation": explanation_full,
                        "payload": shap_payload,
                    }
                )
                st.caption(
                    f"target={prediction_row.get('explanation_target')} · "
                    f"value={prediction_row.get('target_value')}"
                )
                st.dataframe(shap_summary_dataframe(shap_payload), use_container_width=True, hide_index=True)
                components.html(
                    shap_diverging_chart_html(explanation),
                    height=max(360, min(560, 92 + len(explanation) * 42)),
                    scrolling=False,
                )
                with st.expander("Detailed SHAP table"):
                    st.dataframe(
                        explanation_full[["feature", "input_value", "shap_value", "direction"]],
                        use_container_width=True,
                        hide_index=True,
                    )
            except Exception as exc:  # noqa: BLE001
                st.error(f"SHAP 계산 실패: {exc}")

    comparison_df = shap_comparison_dataframe(explanation_payloads)
    if not comparison_df.empty:
        st.markdown("#### Model Difference")
        st.caption("두 모델에서 feature별 SHAP 기여도가 얼마나 다르게 나타나는지 비교합니다.")
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)


def input_summary(values: dict[str, Any]) -> str:
    safe_values = {key: str(value)[:80] for key, value in values.items()}
    return json.dumps(safe_values, ensure_ascii=False, sort_keys=True)


def render_metric_summary(record: dict[str, Any], metrics: dict[str, Any]) -> None:
    metrics_df = metric_comparison_dataframe(record)
    if metrics_df.empty:
        st.info("validation/test metric이 없습니다.")
    else:
        st.markdown("#### Validation vs Test Metrics")
        chart_source = metric_comparison_long(metrics_df)
        if not chart_source.empty:
            chart = (
                alt.Chart(chart_source)
                .mark_bar()
                .encode(
                    x=alt.X("metric:N", title="Metric", sort=None),
                    y=alt.Y("value:Q", title="Value", scale=alt.Scale(zero=False)),
                    color=alt.Color(
                        "split:N",
                        title="Split",
                        scale=alt.Scale(domain=["validation", "test"], range=["#176B87", "#D97706"]),
                    ),
                    xOffset="split:N",
                    tooltip=[
                        alt.Tooltip("metric:N", title="metric"),
                        alt.Tooltip("split:N", title="split"),
                        alt.Tooltip("value:Q", title="value", format=".4f"),
                        alt.Tooltip("gap:Q", title="gap", format=".4f"),
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)
        display_df = metrics_df.copy()
        for column in ["validation", "test", "gap"]:
            if column in display_df.columns:
                display_df[column] = pd.to_numeric(display_df[column], errors="coerce").round(4)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    report_df = class_report_dataframe(metrics)
    if report_df.empty:
        st.info("validation classification_report가 없습니다.")
    else:
        st.markdown("#### Validation Classification Report")
        st.dataframe(report_df, use_container_width=True, hide_index=True)


def split_summary_text(split: dict[str, Any]) -> str:
    return (
        f"train={split.get('train_rows', 'n/a')} · "
        f"validation={split.get('val_rows', 'n/a')} · "
        f"test={split.get('test_rows', 'n/a')}"
    )


def render_experiment_conditions(
    record: dict[str, Any],
    summary: dict[str, Any],
    config: dict[str, Any],
    target: str,
    primary_metric: str,
) -> None:
    split = summary.get("split") if summary else {}
    if not split:
        split = config.get("data", {}).get("split", {}) if isinstance(config.get("data"), dict) else {}
    rows = [
        {"item": "Data version", "value": display_data_name(record.get("data_version"))},
        {"item": "Target", "value": target},
        {"item": "Task", "value": str(record.get("task_type") or summary.get("task_type") or "classification")},
        {"item": "Primary metric", "value": f"{primary_metric} ({metric_direction_label(primary_metric)})"},
        {"item": "Split", "value": split_summary_text(split)},
    ]
    if summary:
        rows.extend(
            [
                {"item": "AutoML backend", "value": str(summary.get("backend", "n/a"))},
                {"item": "Preset / time limit", "value": f"{summary.get('presets', 'n/a')} / {summary.get('time_limit', 'n/a')}s"},
            ]
        )
    st.markdown("#### Experiment Conditions")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_automl_tradeoff(
    record: dict[str, Any],
    primary_metric: str,
    leaderboard: pd.DataFrame,
    selected_model: str,
) -> None:
    if leaderboard.empty:
        missing = missing_artifact_labels(record, [("leaderboard", "leaderboard_path")])
        message = "AutoML 후보 leaderboard를 찾을 수 없습니다."
        if missing:
            message += " " + " / ".join(missing)
        st.warning(message)
        return

    selected_row = selected_leaderboard_row(leaderboard, selected_model)
    best_single = best_single_model_row(leaderboard)
    if selected_row is None:
        st.warning("선택된 AutoML model row를 leaderboard에서 찾지 못했습니다.")
        return

    selected_val = numeric_row_value(selected_row, "score_val")
    selected_test = numeric_row_value(selected_row, "score_test")
    selected_time = prediction_time_value(selected_row)
    single_name = str(best_single.get("model")) if best_single is not None else None
    single_val = numeric_row_value(best_single, "score_val") if best_single is not None else None
    single_test = numeric_row_value(best_single, "score_test") if best_single is not None else None
    single_time = prediction_time_value(best_single)
    validation_gain = metric_delta_value(selected_val, single_val, primary_metric)
    test_gain = metric_delta_value(selected_test, single_test, primary_metric)
    speed_ratio = selected_time / single_time if selected_time is not None and single_time and single_time > 0 else None

    st.markdown("#### Ensemble Tradeoff")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected", display_model_name(record, selected_model))
    c2.metric("Best single", display_model_name(record, single_name) if single_name else "n/a")
    c3.metric("Validation gain", format_optional_delta(validation_gain))
    c4.metric("Speed cost", format_multiplier(speed_ratio))

    if is_ensemble_model_name(selected_model):
        if test_gain is not None and test_gain < 0:
            st.warning(
                "앙상블은 validation 기준으로 선택됐지만 test metric에서는 best single-model보다 낮습니다. "
                f"리뷰/배포 설명에서는 {display_model_name(record, single_name)} 대안을 함께 제시하는 편이 좋습니다."
            )
        elif speed_ratio is not None and speed_ratio > 5 and (validation_gain is None or validation_gain < 0.005):
            st.warning(
                "앙상블의 validation 이득이 작고 예측 비용이 큽니다. "
                "성능 차이가 과제 기준에서 중요한지 확인해야 합니다."
            )
        else:
            st.success("앙상블 선택 근거가 있습니다. validation 이득과 test 안정성을 함께 설명하세요.")
    else:
        st.info("선택 모델이 단일 모델입니다. latency와 재현성 설명이 비교적 단순합니다.")

    detail_rows = [
        {"item": "Selected validation", "value": format_metric(selected_val)},
        {"item": "Selected test", "value": format_metric(selected_test)},
        {"item": "Best single validation", "value": format_metric(single_val)},
        {"item": "Best single test", "value": format_metric(single_test)},
        {"item": "Test delta vs best single", "value": format_optional_delta(test_gain)},
        {"item": "Selected prediction time", "value": format_seconds(selected_time)},
        {"item": "Best single prediction time", "value": format_seconds(single_time)},
    ]
    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)


def render_decision_tab(
    record: dict[str, Any],
    best_record: dict[str, Any] | None,
    baseline: dict[str, Any] | None,
    primary_metric: str,
) -> None:
    st.subheader("Decision Snapshot")
    config = load_run_config(record.get("experiment_path", ""))
    data_config = run_data_config(config)
    data = load_training_data(data_config.get("path"))
    target = target_column(config) or data_config.get("target_column") or "n/a"
    summary = load_automl_summary(record.get("automl_summary_path"))
    leaderboard = load_csv_artifact(record.get("leaderboard_path"))
    selected_model = selected_automl_model(record, summary) if is_automl_record(record) else str(record.get("model_name") or record.get("model_id"))
    selected_row = selected_leaderboard_row(leaderboard, selected_model)
    best_single = best_single_model_row(leaderboard)
    best_single_name = str(best_single.get("model")) if best_single is not None else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recommended model", display_model_name(record, selected_model))
    c2.metric("Best single alternative", display_model_name(record, best_single_name) if best_single_name else "n/a")
    c3.metric("Δ vs baseline", format_delta(metric_delta(record, baseline, primary_metric)))
    c4.metric("Validation-test gap", format_metric(validation_test_gap(record, primary_metric)))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Validation {primary_metric}", format_metric(validation_metric_value(record, primary_metric)))
    m2.metric(f"Test {primary_metric}", format_metric(test_metric_value(record, primary_metric)))
    m3.metric("Prediction time", format_seconds(prediction_time_value(selected_row)))
    m4.metric("Experiment type", display_experiment_type(record))

    if same_record(record, best_record):
        st.success(f"Current best by validation {primary_metric}.")
    elif best_record is not None:
        st.info(f"Current best: {display_model_name(best_record)}")

    if is_automl_record(record):
        render_automl_tradeoff(record, primary_metric, leaderboard, selected_model)

    render_experiment_conditions(record, summary, config, str(target), primary_metric)

    critical_missing = missing_artifact_labels(
        record,
        [("model artifact", "artifact_path"), ("run directory", "experiment_path")],
    )
    if critical_missing:
        st.warning("선택한 run의 일부 artifact가 없습니다. " + " / ".join(critical_missing))

    render_run_source_diagnostics(record)

    if not data.empty:
        feature_columns = [column for column in data.columns if column != target]
        numeric_count = int(data[feature_columns].select_dtypes(include=["number"]).shape[1]) if feature_columns else 0
        categorical_count = max(len(feature_columns) - numeric_count, 0)
        missing_rate = float(data[feature_columns].isna().mean().mean()) if feature_columns else 0.0
        with st.expander("Data Profile"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Features", len(feature_columns))
            c2.metric("Numeric / Categorical", f"{numeric_count} / {categorical_count}")
            c3.metric("Mean Missing Rate", format_percent(missing_rate))

            if target in data.columns and str(record.get("task_type") or "classification") == "classification":
                distribution = data[target].value_counts(dropna=False).rename_axis("class").reset_index(name="count")
                st.dataframe(distribution, use_container_width=True, hide_index=True)

    with st.expander("Full Run Metadata"):
        st.write(f"model_id: `{record.get('model_id')}`")
        st.write(f"run_id: `{record.get('run_id')}`")
        st.write(f"data_version: `{record.get('data_version')}`")
        st.write(f"artifact_path: `{record.get('artifact_path')}`")
        st.write(f"experiment_path: `{record.get('experiment_path')}`")
        if summary:
            st.markdown("##### Raw AutoGluon summary JSON")
            st.json(summary, expanded=False)


def render_leaderboard_tab(
    registry: list[dict[str, Any]],
    record: dict[str, Any],
    best_record: dict[str, Any] | None,
    baseline: dict[str, Any] | None,
    primary_metric: str,
) -> None:
    title = "AutoGluon Candidates" if is_automl_record(record) else "Model Comparison"
    st.subheader(title)
    leaderboard = load_csv_artifact(record.get("leaderboard_path"))
    if not leaderboard.empty:
        summary = load_automl_summary(record.get("automl_summary_path"))
        selected_model = selected_automl_model(record, summary)
        selected_row = selected_leaderboard_row(leaderboard, selected_model)
        candidates = automl_candidate_dataframe(leaderboard, selected_model, primary_metric)

        if selected_row is not None:
            st.markdown(f"**Selected model:** {display_model_name(record, selected_model)}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Candidates", len(leaderboard))
            c2.metric("Validation Score", format_metric(selected_row.get("score_val")))
            c3.metric("Test Score", format_metric(selected_row.get("score_test")))
            c4.metric("Prediction Time", f"{prediction_time_value(selected_row):.4f}s" if prediction_time_value(selected_row) is not None else "n/a")

        st.caption(
            "후보 pipeline을 성능, 안정성, 예측 비용 기준으로 비교합니다. "
            f"현재 primary metric은 {primary_metric}이며 {metric_direction_label(primary_metric)}입니다."
        )
        if not candidates.empty:
            filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 2])
            top_n = filter_col1.slider(
                "Top N",
                min_value=1,
                max_value=max(1, min(25, len(candidates))),
                value=min(10, len(candidates)),
                step=1,
                key=f"candidate_top_n_{record.get('run_id')}",
            )
            hide_ensembles = filter_col2.checkbox(
                "Exclude ensembles",
                value=False,
                key=f"candidate_hide_ensembles_{record.get('run_id')}",
            )
            fast_only = filter_col3.checkbox(
                "Fast models only (≤ selected prediction time)",
                value=False,
                key=f"candidate_fast_only_{record.get('run_id')}",
            )

            filtered = candidates.copy()
            if hide_ensembles and "is_ensemble" in filtered.columns:
                filtered = filtered[~filtered["is_ensemble"]]
            selected_time = prediction_time_value(selected_row)
            if fast_only and selected_time is not None and "prediction_time_s" in filtered.columns:
                filtered = filtered[pd.to_numeric(filtered["prediction_time_s"], errors="coerce") <= selected_time]
            filtered = filtered.sort_values("validation_rank", na_position="last").head(top_n)

            chart_df = candidate_chart_df(filtered)
            chart_df = chart_df.dropna(subset=["prediction_time_s", "validation_score"])
            if not chart_df.empty:
                st.markdown("#### Score vs Speed Tradeoff")
                score_choice = st.radio(
                    "Score axis",
                    options=["validation_score", "test_score"],
                    index=0,
                    horizontal=True,
                    key=f"score_axis_{record.get('run_id')}",
                )
                scatter_source = chart_df.dropna(subset=[score_choice, "prediction_time_s"]).copy()
                scatter = (
                    alt.Chart(scatter_source)
                    .mark_circle(opacity=0.85)
                    .encode(
                        x=alt.X("prediction_time_s:Q", title="Prediction time (s)", scale=alt.Scale(zero=False)),
                        y=alt.Y(f"{score_choice}:Q", title=score_choice.replace("_", " "), scale=alt.Scale(zero=False)),
                        color=alt.Color("role:N", scale=role_color_scale(), title="Role"),
                        size=alt.Size("role:N", scale=alt.Scale(domain=["Selected", "Best single model", "Candidate"], range=[220, 160, 90]), legend=None),
                        tooltip=[
                            alt.Tooltip("model:N", title="model"),
                            alt.Tooltip("role:N", title="role"),
                            alt.Tooltip("validation_score:Q", title="validation", format=".4f"),
                            alt.Tooltip("test_score:Q", title="test", format=".4f"),
                            alt.Tooltip("validation_test_gap:Q", title="gap", format=".4f"),
                            alt.Tooltip("prediction_time_s:Q", title="prediction s", format=".4f"),
                        ],
                    )
                    .properties(height=320)
                )
                st.altair_chart(scatter, use_container_width=True)

                st.markdown("#### Validation-Test Stability")
                stability_source = chart_df.dropna(subset=["validation_score", "test_score"]).copy()
                stability_long = stability_source.melt(
                    id_vars=["model", "role", "validation_test_gap"],
                    value_vars=["validation_score", "test_score"],
                    var_name="split",
                    value_name="score",
                )
                stability_long["split"] = stability_long["split"].map(
                    {"validation_score": "validation", "test_score": "test"}
                )
                stability = (
                    alt.Chart(stability_long)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("split:N", sort=["validation", "test"], title="Split"),
                        y=alt.Y("score:Q", title=primary_metric, scale=alt.Scale(zero=False)),
                        detail="model:N",
                        color=alt.Color("role:N", scale=role_color_scale(), title="Role"),
                        tooltip=[
                            alt.Tooltip("model:N", title="model"),
                            alt.Tooltip("role:N", title="role"),
                            alt.Tooltip("split:N", title="split"),
                            alt.Tooltip("score:Q", title="score", format=".4f"),
                            alt.Tooltip("validation_test_gap:Q", title="gap", format=".4f"),
                        ],
                    )
                    .properties(height=300)
                )
                st.altair_chart(stability, use_container_width=True)
            else:
                st.info("필터 조건에 맞는 후보가 없습니다.")

            st.markdown("#### Candidate Table")
            display_df = candidate_display_dataframe(filtered)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        with st.expander("Full AutoGluon leaderboard"):
            st.dataframe(leaderboard, use_container_width=True, hide_index=True)
        render_run_source_diagnostics(record)
        return

    if is_automl_record(record):
        missing = missing_artifact_labels(record, [("leaderboard", "leaderboard_path")])
        if missing:
            st.warning("AutoML 후보 비교 파일을 찾을 수 없습니다. " + " / ".join(missing))
        else:
            st.warning("AutoML 후보 leaderboard가 비어 있습니다.")
        render_run_source_diagnostics(record)
        return

    registry_df = registry_to_dataframe(registry, primary_metric, best_record, baseline)
    if registry_df.empty:
        st.info("비교할 registry record가 없습니다.")
        return

    st.dataframe(registry_df, use_container_width=True, hide_index=True)
    trend_source = registry_trend_dataframe(registry)
    trend_cols = [col for col in ["val_accuracy", "val_macro_f1"] if col in trend_source.columns]
    if trend_cols and len(trend_source) > 1:
        st.line_chart(trend_source.set_index("created_at")[trend_cols])


def render_threshold_section(record: dict[str, Any]) -> None:
    threshold_df = load_csv_artifact(record.get("threshold_metrics_path"))
    if threshold_df.empty:
        missing = missing_artifact_labels(record, [("threshold metrics", "threshold_metrics_path")])
        if missing:
            st.info("threshold tuning 파일을 찾을 수 없습니다. " + " / ".join(missing))
        return

    st.subheader("Threshold Tuning (Validation)")
    curve_df = threshold_curve_dataframe(threshold_df)
    if not curve_df.empty:
        chart = (
            alt.Chart(curve_df)
            .mark_line()
            .encode(
                x=alt.X("threshold:Q", title="Threshold", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("value:Q", title="Metric value", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "metric:N",
                    title="Metric",
                    scale=alt.Scale(domain=["precision", "recall", "f1"], range=["#176B87", "#D97706", "#16A34A"]),
                ),
                tooltip=[
                    alt.Tooltip("threshold:Q", title="threshold", format=".2f"),
                    alt.Tooltip("metric:N", title="metric"),
                    alt.Tooltip("value:Q", title="value", format=".4f"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

    threshold_values = threshold_df["threshold"].astype(float).tolist()
    selected_threshold = st.slider("Threshold", 0.0, 1.0, 0.5, 0.01, key="evaluation_threshold")
    nearest_threshold = min(threshold_values, key=lambda value: abs(value - selected_threshold))
    threshold_float = threshold_df["threshold"].astype(float)
    row = threshold_df.loc[(threshold_float - nearest_threshold).abs().idxmin()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", format_metric(row.get("precision")))
    c2.metric("Recall", format_metric(row.get("recall")))
    c3.metric("F1", format_metric(row.get("f1")))
    c4.metric("Threshold", f"{float(row.get('threshold')):.2f}")

    count_cols = st.columns(4)
    for column, label, help_text in [
        ("tp", "TP", "actual positive, predicted positive"),
        ("fp", "FP", "actual negative, predicted positive"),
        ("tn", "TN", "actual negative, predicted negative"),
        ("fn", "FN", "actual positive, predicted negative"),
    ]:
        value = int(row.get(column, 0))
        count_cols[["tp", "fp", "tn", "fn"].index(column)].metric(label, value, help=help_text)

    with st.expander("All threshold metrics"):
        st.dataframe(threshold_df, use_container_width=True, hide_index=True)


def render_evaluation_tab(record: dict[str, Any]) -> None:
    st.subheader("Evaluation")
    render_run_source_diagnostics(record)

    metrics = load_metrics(record.get("experiment_path", ""))
    render_metric_summary(record, metrics)

    if record.get("task_type", "classification") != "classification":
        st.info("Regression run은 confusion matrix와 threshold tuning을 표시하지 않습니다.")
        return

    render_threshold_section(record)

    st.subheader("Validation Confusion Matrix")
    confusion = load_confusion_matrix(record.get("confusion_matrix_path", ""))
    cm_long = confusion_matrix_long_dataframe(confusion)
    if cm_long.empty:
        missing = missing_artifact_labels(record, [("confusion matrix", "confusion_matrix_path")])
        if missing:
            st.info("confusion matrix 파일을 찾을 수 없습니다. " + " / ".join(missing))
        else:
            st.info("confusion_matrix.json을 찾을 수 없습니다.")
    else:
        heatmap = (
            alt.Chart(cm_long)
            .mark_rect()
            .encode(
                x=alt.X("predicted:N", title="Predicted"),
                y=alt.Y("actual:N", title="Actual"),
                color=alt.Color("count:Q", title="Count", scale=alt.Scale(scheme="blues")),
                tooltip=[
                    alt.Tooltip("actual:N", title="actual"),
                    alt.Tooltip("predicted:N", title="predicted"),
                    alt.Tooltip("count:Q", title="count"),
                    alt.Tooltip("row_percent:Q", title="row %", format=".1%"),
                ],
            )
            .properties(height=260)
        )
        text = (
            alt.Chart(cm_long)
            .mark_text(baseline="middle")
            .encode(
                x=alt.X("predicted:N"),
                y=alt.Y("actual:N"),
                text=alt.Text("count:Q"),
                color=alt.condition(alt.datum.row_percent > 0.45, alt.value("white"), alt.value("#111827")),
            )
        )
        st.altair_chart(heatmap + text, use_container_width=True)
        cm_df = confusion_matrix_dataframe(confusion)
        with st.expander("Confusion matrix table"):
            st.dataframe(cm_df, use_container_width=True)
        st.caption(confusion.get("format", "rows=true_label, columns=predicted_label"))


def render_predict_tab(
    record: dict[str, Any],
    best_record: dict[str, Any] | None,
    primary_metric: str,
) -> None:
    st.subheader("Sample Prediction")

    if same_record(record, best_record):
        st.success(
            f"Current champion · validation {primary_metric}="
            f"{format_metric(validation_metric_value(record, primary_metric))}"
        )
    elif best_record is not None:
        st.warning(
            f"Not current best · best is {short_id(best_record.get('model_id'))} "
            f"(validation {primary_metric}={format_metric(validation_metric_value(best_record, primary_metric))})"
        )

    artifact_path = record.get("artifact_path")
    if not artifact_path:
        st.error("선택한 모델 record에 artifact_path가 없습니다.")
        return
    resolved_artifact = resolve_path(artifact_path)
    if resolved_artifact is None or not resolved_artifact.exists():
        st.error(f"모델 파일을 찾을 수 없습니다: {artifact_path}")
        return

    config = load_run_config(record.get("experiment_path", ""))
    data_config = run_data_config(config)
    training_data = load_training_data(data_config.get("path"))

    try:
        model = load_autogluon_model(artifact_path) if is_automl_record(record) else load_model(artifact_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"모델 로드 실패: {exc}")
        return

    feature_columns = model_feature_columns(model, config, training_data)
    if not feature_columns:
        st.info("feature 목록을 찾지 못했습니다. run config와 processed CSV를 확인하세요.")
        return

    total_features, numeric_features, categorical_features = feature_counts(feature_columns, training_data)
    c1, c2, c3 = st.columns(3)
    c1.metric("Features", total_features)
    c2.metric("Numeric", numeric_features)
    c3.metric("Categorical", categorical_features)

    prediction_model_name = None
    if is_automl_record(record):
        summary = load_automl_summary(record.get("automl_summary_path"))
        leaderboard = load_csv_artifact(record.get("leaderboard_path"))
        selected_model = selected_automl_model(record, summary)
        model_options = automl_model_options(model, leaderboard, selected_model)
        if model_options:
            default_index = model_options.index(selected_model) if selected_model in model_options else 0
            prediction_model_name = st.selectbox(
                "Prediction model",
                options=model_options,
                index=default_index,
                format_func=lambda name: display_model_name(record, name),
                key=f"prediction_model_{record.get('run_id')}",
                help=(
                    "기본값은 AutoGluon이 validation metric으로 선택한 모델입니다. "
                    "WeightedEnsemble 계열은 내부 base model을 함께 호출할 수 있어 더 느릴 수 있습니다."
                ),
            )
            selected_row = selected_leaderboard_row(leaderboard, prediction_model_name)
            if selected_row is not None:
                st.caption(
                    f"raw_model={prediction_model_name} · "
                    f"validation_score={format_metric(selected_row.get('score_val'))} · "
                    f"test_score={format_metric(selected_row.get('score_test'))}"
                )
        else:
            st.info("AutoGluon 내부 모델 목록을 찾지 못해 기본 predictor로 예측합니다.")

    with st.expander("Feature columns"):
        st.dataframe(feature_details_dataframe(feature_columns, training_data), use_container_width=True, hide_index=True)

    with st.form("prediction_form"):
        input_cols = st.columns(2)
        values = {}
        for index, column in enumerate(feature_columns):
            with input_cols[index % 2]:
                values[column] = render_feature_input(column, training_data)
        submitted = st.form_submit_button("Run prediction", use_container_width=True)

    prediction_state_key = f"prediction_values_{record.get('run_id')}"
    if submitted:
        st.session_state[prediction_state_key] = values
    active_values = st.session_state.get(prediction_state_key)

    if active_values is None:
        return

    logger = InferenceLogger(log_path=str(LOG_PATH))
    start = time.perf_counter()
    summary = input_summary(active_values)

    try:
        features = pd.DataFrame([active_values], columns=feature_columns)
        prediction = prediction_value(model, features, prediction_model_name)
        probabilities = prediction_probabilities(model, features, prediction_model_name)
        latency_ms = (time.perf_counter() - start) * 1000

        prediction_key = str(prediction)
        confidence = probabilities.get(prediction_key) if probabilities else None

        st.divider()
        st.subheader("Prediction Result")
        if not submitted:
            st.caption("마지막으로 제출한 입력값 기준 결과입니다.")
        result_col1, result_col2, result_col3 = st.columns(3)
        result_col1.metric("Predicted label", prediction_key)
        result_col2.metric("Confidence", format_percent(confidence))
        result_col3.metric("Latency", f"{latency_ms:.1f} ms")
        if prediction_model_name:
            st.markdown(f"**Prediction model:** {display_model_name(record, prediction_model_name)}")
            st.caption(f"Raw AutoGluon model: `{prediction_model_name}`")

        st.markdown("#### Class Probabilities")
        if probabilities:
            probability_df = pd.DataFrame(
                [
                    {
                        "class": label,
                        "probability": probability,
                        "probability_%": format_percent(probability),
                    }
                    for label, probability in probabilities.items()
                ]
            )
            st.dataframe(probability_df[["class", "probability_%"]], use_container_width=True, hide_index=True)
            st.bar_chart(probability_df.set_index("class")[["probability"]])

            threshold_df = load_csv_artifact(record.get("threshold_metrics_path"))
            positive_class = str(record.get("positive_class") or "")
            if not threshold_df.empty and positive_class in probabilities:
                threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
                negative_class = next((label for label in probabilities if label != positive_class), prediction_key)
                threshold_prediction = positive_class if probabilities[positive_class] >= threshold else negative_class
                st.caption(
                    f"threshold={threshold:.2f} 기준 label={threshold_prediction} "
                    f"(positive probability={format_percent(probabilities[positive_class])})"
                )
        else:
            st.info("classification probability를 제공하지 않는 모델입니다.")
        st.caption(f"model_id={short_id(record.get('model_id'))} / run_id={short_id(record.get('run_id'))}")

        render_autogluon_local_explanation(
            record=record,
            predictor=model,
            config=config,
            training_data=training_data,
            feature_columns=feature_columns,
            features=features,
        )

        if submitted:
            logger.log(
                input_summary=summary,
                prediction=prediction_key,
                latency_ms=latency_ms,
                model_id=record.get("model_id"),
                run_id=record.get("run_id"),
                autogluon_model=prediction_model_name,
                probability=round(probabilities.get(prediction_key), 6) if probabilities else None,
                error_message="",
            )
            try:
                load_logs.clear()
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - start) * 1000
        if submitted:
            logger.log(
                input_summary=summary,
                prediction=None,
                latency_ms=latency_ms,
                status="error",
                model_id=record.get("model_id"),
                run_id=record.get("run_id"),
                probability=None,
                error_message=str(exc),
            )
        st.error(f"예측 실패: {exc}")


def render_logs_tab() -> None:
    st.subheader("Inference Logs")
    logs = load_logs()
    if logs.empty:
        st.info("아직 추론 로그가 없습니다. Prediction 탭에서 예측을 실행하면 logs/inference.jsonl에 기록됩니다.")
        return

    total = len(logs)
    success = int((logs.get("status") == "success").sum()) if "status" in logs else 0
    errors = int((logs.get("status") == "error").sum()) if "status" in logs else 0
    error_rate = errors / total if total else 0
    mean_latency = logs["latency_ms"].mean() if "latency_ms" in logs else 0
    p95_latency = logs["latency_ms"].quantile(0.95) if "latency_ms" in logs else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Requests", total)
    col2.metric("Success", success)
    col3.metric("Errors", errors)
    col4.metric("Error Rate", f"{error_rate:.1%}")
    col5.metric("P95 Latency", f"{p95_latency:.1f} ms")
    st.caption(f"Average latency: {mean_latency:.1f} ms")

    display_columns = [
        column
        for column in [
            "timestamp",
            "status",
            "prediction",
            "probability",
            "latency_ms",
            "model_id",
            "run_id",
            "input_summary",
            "error_message",
        ]
        if column in logs.columns
    ]
    display_logs = logs[display_columns].head(50).copy()
    if "probability" in display_logs.columns:
        display_logs["probability"] = display_logs["probability"].apply(format_percent)
    if "model_id" in display_logs.columns:
        display_logs["model_id"] = display_logs["model_id"].apply(short_id)
    if "run_id" in display_logs.columns:
        display_logs["run_id"] = display_logs["run_id"].apply(short_id)
    if "input_summary" in display_logs.columns:
        display_logs["input_summary"] = display_logs["input_summary"].apply(lambda value: truncate_text(value, 72))

    st.dataframe(display_logs, use_container_width=True, hide_index=True)

    if "input_summary" in logs.columns:
        with st.expander("Full input summaries"):
            detail_columns = [
                column
                for column in ["timestamp", "status", "prediction", "model_id", "run_id", "input_summary"]
                if column in logs.columns
            ]
            st.dataframe(logs[detail_columns].head(50), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="ML Experiment Dashboard", layout="wide")
    st.title("ML Experiment Dashboard")
    st.caption("baseline과 AutoGluon Tabular 실험의 metric, leaderboard, threshold, 예측 로그를 확인합니다.")

    registry = load_registry()
    if not registry:
        st.warning("model_registry.json이 비어 있습니다. 먼저 /train 또는 scripts/train.py를 실행하세요.")
        with st.expander("Run Source"):
            st.write(f"BASE_DIR: `{BASE_DIR}`")
            st.write(f"registry: `{REGISTRY_PATH}`")
            st.write(f"registry_exists: `{REGISTRY_PATH.exists()}`")
        return

    default_record = registry[default_selected_index(registry)]
    data_options = data_versions_from_registry(registry)
    default_data_version = default_record.get("data_version")
    data_index = data_options.index(default_data_version) if default_data_version in data_options else 0
    selected_data_version = st.sidebar.selectbox(
        "Data",
        options=data_options,
        index=data_index,
        format_func=display_data_name,
    )

    data_records = records_for_data_version(registry, selected_data_version)
    if not data_records:
        st.warning("선택한 data version에 연결된 model record가 없습니다.")
        return

    sidebar_metric = str((data_records[-1]).get("primary_metric") or "macro_f1")
    model_options = ranked_records(data_records, sidebar_metric) or data_records
    preferred_record = best_model_record(data_records, selected_data_version, sidebar_metric) or model_options[0]
    model_index = next(
        (index for index, record in enumerate(model_options) if same_record(record, preferred_record)),
        0,
    )
    selected_model_index = st.sidebar.selectbox(
        "Model",
        options=list(range(len(model_options))),
        index=model_index,
        format_func=lambda index: display_run_label(model_options[index]),
    )
    selected_record = model_options[selected_model_index]
    primary_metric = str(selected_record.get("primary_metric") or "macro_f1")
    comparison_records = records_for_data_version(registry, selected_record.get("data_version"))
    best_record = best_model_record(registry, selected_record.get("data_version"), primary_metric)
    baseline = baseline_record(comparison_records)

    st.sidebar.markdown("### Best Model")
    if best_record is not None:
        best_value = validation_metric_value(best_record, primary_metric)
        best_delta = metric_delta(best_record, baseline, primary_metric)
        st.sidebar.success(display_model_name(best_record))
        st.sidebar.caption(
            f"validation {primary_metric}={format_metric(best_value)}"
            f" · Δ vs baseline={format_delta(best_delta)}"
        )
    else:
        st.sidebar.info("validation metric이 있는 모델이 없습니다.")

    st.sidebar.markdown("### Selected Metrics")
    val_metrics = selected_record.get("metrics", {})
    test_metrics = selected_record.get("test_metrics", {})
    st.sidebar.metric(f"Val {primary_metric}", format_metric(val_metrics.get(primary_metric)))
    st.sidebar.metric(f"Test {primary_metric}", format_metric(test_metrics.get(primary_metric)))
    st.sidebar.caption(f"Data: {display_data_name(selected_record.get('data_version'))}")

    with st.sidebar.expander("Full IDs"):
        st.write(f"model_id: `{selected_record.get('model_id')}`")
        st.write(f"run_id: `{selected_record.get('run_id')}`")
        st.write(f"data_version: `{selected_record.get('data_version')}`")
        st.write(f"primary_metric: `{selected_record.get('primary_metric')}`")

    decision_tab, candidates_tab, evaluation_tab, prediction_tab, logs_tab = st.tabs(
        ["Decision", "Candidates", "Evaluation", "Prediction", "Logs"]
    )
    with decision_tab:
        render_decision_tab(selected_record, best_record, baseline, primary_metric)
    with candidates_tab:
        render_leaderboard_tab(comparison_records, selected_record, best_record, baseline, primary_metric)
    with evaluation_tab:
        render_evaluation_tab(selected_record)
    with prediction_tab:
        render_predict_tab(selected_record, best_record, primary_metric)
    with logs_tab:
        render_logs_tab()


if __name__ == "__main__":
    main()
