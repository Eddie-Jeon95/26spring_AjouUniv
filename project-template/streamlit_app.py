"""
Generic tabular model Streamlit app.

실험 registry, run artifact, 모델 파일, 추론 로그를 한 화면에서 확인합니다.
실행:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml


BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.utils.logger import InferenceLogger  # noqa: E402


REGISTRY_PATH = BASE_DIR / "model_registry.json"
LOG_PATH = BASE_DIR / "logs" / "inference.jsonl"
MISSING_OPTION = "(missing)"


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


def validation_metric_value(record: dict[str, Any], metric_name: str) -> float | None:
    value = record.get("metrics", {}).get(metric_name)
    if isinstance(value, (int, float)) and pd.notna(value):
        return float(value)
    return None


def records_for_data_version(registry: list[dict[str, Any]], data_version: str | None) -> list[dict[str, Any]]:
    if data_version is None:
        return list(registry)
    return [record for record in registry if record.get("data_version") == data_version]


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
        reverse=True,
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
    return value - baseline_value


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


@st.cache_resource
def load_model(artifact_path: str) -> Any:
    resolved = resolve_path(artifact_path)
    if resolved is None:
        raise FileNotFoundError("artifact_path가 비어 있습니다.")
    with open(resolved, "rb") as f:
        return pickle.load(f)


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
                "model_id": short_id(record.get("model_id")),
                "run_id": short_id(record.get("run_id")),
                "data_version": record.get("data_version"),
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
    return f"{short_id(record.get('model_id'))} · {record.get('data_version')} · val {primary_metric}={metric_text}"


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


def prediction_probabilities(model: Any, features: pd.DataFrame) -> dict[str, float]:
    if not hasattr(model, "predict_proba"):
        return {}
    probabilities = model.predict_proba(features)[0]
    classes = [str(value) for value in getattr(model, "classes_", [])]
    return {class_label: float(probabilities[index]) for index, class_label in enumerate(classes)}


def input_summary(values: dict[str, Any]) -> str:
    safe_values = {key: str(value)[:80] for key, value in values.items()}
    return json.dumps(safe_values, ensure_ascii=False, sort_keys=True)


def render_metric_summary(record: dict[str, Any], metrics: dict[str, Any]) -> None:
    metric_values = record.get("metrics", {})
    test_metric_values = record.get("test_metrics", {})
    val_col, test_col = st.columns(2)

    with val_col:
        st.markdown("#### Model Selection Metric (Validation)")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", format_metric(metric_values.get("accuracy")))
        c2.metric("Macro F1", format_metric(metric_values.get("macro_f1")))

    with test_col:
        st.markdown("#### Final Check Metric (Test)")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", format_metric(test_metric_values.get("accuracy")))
        c2.metric("Macro F1", format_metric(test_metric_values.get("macro_f1")))

    report_df = class_report_dataframe(metrics)
    if report_df.empty:
        st.info("validation classification_report가 없습니다.")
    else:
        st.markdown("#### Validation Classification Report")
        st.dataframe(report_df, use_container_width=True, hide_index=True)


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
        model = load_model(artifact_path)
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

    with st.expander("Feature columns"):
        st.dataframe(feature_details_dataframe(feature_columns, training_data), use_container_width=True, hide_index=True)

    with st.form("prediction_form"):
        input_cols = st.columns(2)
        values = {}
        for index, column in enumerate(feature_columns):
            with input_cols[index % 2]:
                values[column] = render_feature_input(column, training_data)
        submitted = st.form_submit_button("Run prediction", use_container_width=True)

    if not submitted:
        return

    logger = InferenceLogger(log_path=str(LOG_PATH))
    start = time.perf_counter()
    summary = input_summary(values)

    try:
        features = pd.DataFrame([values], columns=feature_columns)
        prediction = model.predict(features)[0]
        probabilities = prediction_probabilities(model, features)
        latency_ms = (time.perf_counter() - start) * 1000

        prediction_key = str(prediction)
        confidence = probabilities.get(prediction_key) if probabilities else None

        st.divider()
        st.subheader("Prediction Result")
        result_col1, result_col2, result_col3 = st.columns(3)
        result_col1.metric("Predicted label", prediction_key)
        result_col2.metric("Confidence", format_percent(confidence))
        result_col3.metric("Latency", f"{latency_ms:.1f} ms")

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
        st.caption(f"model_id={short_id(record.get('model_id'))} / run_id={short_id(record.get('run_id'))}")

        logger.log(
            input_summary=summary,
            prediction=prediction_key,
            latency_ms=latency_ms,
            model_id=record.get("model_id"),
            run_id=record.get("run_id"),
            probability=round(probabilities.get(prediction_key), 6) if probabilities else None,
            error_message="",
        )
        st.cache_data.clear()
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - start) * 1000
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


def render_experiments_tab(
    registry: list[dict[str, Any]],
    record: dict[str, Any],
    best_record: dict[str, Any] | None,
    baseline: dict[str, Any] | None,
    primary_metric: str,
) -> None:
    st.subheader(f"Experiment Registry ({record.get('data_version')})")
    registry_df = registry_to_dataframe(registry, primary_metric, best_record, baseline)
    if not registry_df.empty:
        st.dataframe(registry_df, use_container_width=True, hide_index=True)
        trend_source = registry_trend_dataframe(registry)
        trend_cols = [col for col in ["val_accuracy", "val_macro_f1"] if col in trend_source.columns]
        if trend_cols and len(trend_source) > 1:
            trend_df = trend_source.set_index("created_at")[trend_cols]
            st.line_chart(trend_df)

    st.subheader("Selected Run")
    with st.expander("Full run metadata"):
        st.write(f"model_id: `{record.get('model_id')}`")
        st.write(f"run_id: `{record.get('run_id')}`")
        st.write(f"data_version: `{record.get('data_version')}`")
        st.write(f"artifact_path: `{record.get('artifact_path')}`")
        st.write(f"experiment_path: `{record.get('experiment_path')}`")

    metrics = load_metrics(record.get("experiment_path", ""))
    render_metric_summary(record, metrics)

    st.subheader("Validation Confusion Matrix (validation 기준)")
    confusion = load_confusion_matrix(record.get("confusion_matrix_path", ""))
    cm_df = confusion_matrix_dataframe(confusion)
    if cm_df.empty:
        st.info("confusion_matrix.json을 찾을 수 없습니다.")
    else:
        st.dataframe(cm_df, use_container_width=True)
        st.caption(confusion.get("format", "rows=true_label, columns=predicted_label"))


def render_logs_tab() -> None:
    st.subheader("Inference Logs")
    logs = load_logs()
    if logs.empty:
        st.info("아직 추론 로그가 없습니다. Predict 탭에서 예측을 실행하면 logs/inference.jsonl에 기록됩니다.")
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
    st.caption("탭형 데이터 모델의 예측, validation/test metric, 추론 로그를 한 화면에서 확인합니다.")

    registry = load_registry()
    if not registry:
        st.warning("model_registry.json이 비어 있습니다. 먼저 /train-baseline 또는 scripts/train.py를 실행하세요.")
        return

    labels = [model_label(record) for record in registry]
    selected_label = st.sidebar.selectbox("Model", labels, index=default_selected_index(registry))
    selected_index = labels.index(selected_label)
    selected_record = registry[selected_index]
    primary_metric = str(selected_record.get("primary_metric") or "macro_f1")
    comparison_records = records_for_data_version(registry, selected_record.get("data_version"))
    best_record = best_model_record(registry, selected_record.get("data_version"), primary_metric)
    baseline = baseline_record(comparison_records)

    st.sidebar.markdown("### Best Model")
    if best_record is not None:
        best_value = validation_metric_value(best_record, primary_metric)
        best_delta = metric_delta(best_record, baseline, primary_metric)
        st.sidebar.success(f"{short_id(best_record.get('model_id'))}")
        st.sidebar.caption(
            f"validation {primary_metric}={format_metric(best_value)}"
            f" · Δ vs baseline={format_delta(best_delta)}"
        )
    else:
        st.sidebar.info("validation metric이 있는 모델이 없습니다.")

    st.sidebar.markdown("### Selected Metrics")
    val_metrics = selected_record.get("metrics", {})
    test_metrics = selected_record.get("test_metrics", {})
    st.sidebar.metric("Val Macro F1", format_metric(val_metrics.get("macro_f1")))
    st.sidebar.metric("Test Macro F1", format_metric(test_metrics.get("macro_f1")))
    st.sidebar.caption(f"data_version: `{selected_record.get('data_version')}`")

    with st.sidebar.expander("Full IDs"):
        st.write(f"model_id: `{selected_record.get('model_id')}`")
        st.write(f"run_id: `{selected_record.get('run_id')}`")
        st.write(f"primary_metric: `{selected_record.get('primary_metric')}`")

    predict_tab, experiments_tab, logs_tab = st.tabs(["Predict", "Experiments", "Logs"])
    with predict_tab:
        render_predict_tab(selected_record, best_record, primary_metric)
    with experiments_tab:
        render_experiments_tab(comparison_records, selected_record, best_record, baseline, primary_metric)
    with logs_tab:
        render_logs_tab()


if __name__ == "__main__":
    main()
