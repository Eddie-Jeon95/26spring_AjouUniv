"""
Banknote Authentication Streamlit app.

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


BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.utils.logger import InferenceLogger  # noqa: E402


REGISTRY_PATH = BASE_DIR / "model_registry.json"
LOG_PATH = BASE_DIR / "logs" / "inference.jsonl"
FEATURE_COLUMNS = ["variance", "skewness", "curtosis", "entropy"]
DEFAULT_INPUT = {
    "variance": 3.6216,
    "skewness": 8.6661,
    "curtosis": -2.8073,
    "entropy": -0.44699,
}
LABEL_NAMES = {
    "0": "진짜 지폐",
    "1": "위조 지폐",
}


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_registry() -> list[dict[str, Any]]:
    registry = read_json(REGISTRY_PATH, [])
    return registry if isinstance(registry, list) else []


@st.cache_data
def load_metrics(experiment_path: str) -> dict[str, Any]:
    return read_json(BASE_DIR / experiment_path / "metrics.json", {})


@st.cache_data
def load_confusion_matrix(path: str) -> dict[str, Any]:
    return read_json(BASE_DIR / path, {})


@st.cache_resource
def load_model(artifact_path: str) -> Any:
    with open(BASE_DIR / artifact_path, "rb") as f:
        return pickle.load(f)


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


def registry_to_dataframe(registry: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in registry:
        metrics = record.get("metrics", {})
        rows.append(
            {
                "created_at": record.get("created_at"),
                "model_id": record.get("model_id"),
                "run_id": record.get("run_id"),
                "data_version": record.get("data_version"),
                "primary_metric": record.get("primary_metric"),
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
            }
        )
    return pd.DataFrame(rows)


def model_label(record: dict[str, Any]) -> str:
    metric = record.get("metrics", {}).get(record.get("primary_metric", "macro_f1"))
    metric_text = f"{metric:.4f}" if isinstance(metric, (int, float)) else "n/a"
    return f"{record.get('model_id')} | {record.get('data_version')} | {metric_text}"


def class_report_dataframe(metrics: dict[str, Any]) -> pd.DataFrame:
    report = metrics.get("classification_report", {})
    rows = []
    for label in ("0", "1", "macro avg", "weighted avg"):
        values = report.get(label)
        if not isinstance(values, dict):
            continue
        rows.append(
            {
                "class": label,
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


def positive_probability(model: Any, features: pd.DataFrame) -> float | None:
    if not hasattr(model, "predict_proba"):
        return None
    probabilities = model.predict_proba(features)[0]
    classes = [str(value) for value in getattr(model, "classes_", [])]
    if "1" in classes:
        return float(probabilities[classes.index("1")])
    if len(probabilities) > 1:
        return float(probabilities[1])
    return None


def input_summary(values: dict[str, float]) -> str:
    return ", ".join(f"{name}={values[name]:.5g}" for name in FEATURE_COLUMNS)


def render_metric_summary(record: dict[str, Any], metrics: dict[str, Any]) -> None:
    metric_values = record.get("metrics", {})
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metric_values.get('accuracy', 0):.4f}")
    col2.metric("Macro F1", f"{metric_values.get('macro_f1', 0):.4f}")
    col3.metric("Primary", record.get("primary_metric", "-"))

    report_df = class_report_dataframe(metrics)
    if report_df.empty:
        st.info("classification_report가 없습니다.")
    else:
        st.dataframe(report_df, use_container_width=True, hide_index=True)


def model_feature_columns(model: Any) -> list[str]:
    """모델이 학습된 feature 목록을 scaler에서 읽는다. 없으면 전역 기본값 사용."""
    try:
        return list(model.named_steps["scaler"].feature_names_in_)
    except (AttributeError, KeyError):
        return list(FEATURE_COLUMNS)


def render_predict_tab(record: dict[str, Any]) -> None:
    st.subheader("Single Prediction")

    artifact_path = record.get("artifact_path")
    if not artifact_path:
        st.error("선택한 모델 record에 artifact_path가 없습니다.")
        return
    if not (BASE_DIR / artifact_path).exists():
        st.error(f"모델 파일을 찾을 수 없습니다: {artifact_path}")
        return

    try:
        model = load_model(artifact_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"모델 로드 실패: {exc}")
        return

    active_cols = model_feature_columns(model)
    st.caption(f"선택한 모델의 입력 feature: {active_cols}")

    with st.form("prediction_form"):
        values = {
            name: st.number_input(
                name,
                value=float(DEFAULT_INPUT.get(name, 0.0)),
                format="%.5f",
            )
            for name in active_cols
        }
        submitted = st.form_submit_button("예측하기")

    if not submitted:
        return

    logger = InferenceLogger(log_path=str(LOG_PATH))
    start = time.perf_counter()
    summary = input_summary({k: values[k] for k in active_cols if k in values})

    try:
        features = pd.DataFrame([values], columns=active_cols)
        prediction = model.predict(features)[0]
        probability = positive_probability(model, features)
        latency_ms = (time.perf_counter() - start) * 1000

        prediction_key = str(prediction)
        label_text = LABEL_NAMES.get(prediction_key, prediction_key)
        if prediction_key == "1":
            st.error(f"예측: {label_text} ({prediction_key})")
        else:
            st.success(f"예측: {label_text} ({prediction_key})")

        if probability is not None:
            st.metric("위조 확률", f"{probability:.2%}")
        st.caption(f"model_id={record.get('model_id')} / run_id={record.get('run_id')}")

        logger.log(
            input_summary=summary,
            prediction=prediction_key,
            latency_ms=latency_ms,
            model_id=record.get("model_id"),
            run_id=record.get("run_id"),
            probability=round(probability, 6) if probability is not None else None,
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


def render_experiments_tab(registry: list[dict[str, Any]], record: dict[str, Any]) -> None:
    st.subheader("Experiment Registry")
    registry_df = registry_to_dataframe(registry)
    if not registry_df.empty:
        st.dataframe(registry_df, use_container_width=True, hide_index=True)
        trend_cols = [col for col in ["accuracy", "macro_f1"] if col in registry_df.columns]
        if trend_cols and len(registry_df) > 1:
            trend_df = registry_df.set_index("created_at")[trend_cols]
            st.line_chart(trend_df)

    st.subheader("Selected Run")
    metrics = load_metrics(record.get("experiment_path", ""))
    render_metric_summary(record, metrics)

    st.subheader("Confusion Matrix")
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
            "model_id",
            "run_id",
            "status",
            "latency_ms",
            "prediction",
            "probability",
            "error_message",
            "input_summary",
        ]
        if column in logs.columns
    ]
    st.dataframe(logs[display_columns].head(50), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Banknote Authentication Demo", layout="wide")
    st.title("Banknote Authentication Demo")
    st.caption("모델 버전, 예측 결과, 추론 로그를 한 화면에서 확인합니다.")

    registry = load_registry()
    if not registry:
        st.warning("model_registry.json이 비어 있습니다. 먼저 /train-baseline 또는 scripts/train.py를 실행하세요.")
        return

    labels = [model_label(record) for record in registry]
    selected_label = st.sidebar.selectbox("Model / Run", labels, index=len(labels) - 1)
    selected_index = labels.index(selected_label)
    selected_record = registry[selected_index]

    st.sidebar.markdown("### Selected")
    st.sidebar.write(f"model_id: `{selected_record.get('model_id')}`")
    st.sidebar.write(f"run_id: `{selected_record.get('run_id')}`")
    st.sidebar.write(f"data_version: `{selected_record.get('data_version')}`")

    predict_tab, experiments_tab, logs_tab = st.tabs(["Predict", "Experiments", "Logs"])
    with predict_tab:
        render_predict_tab(selected_record)
    with experiments_tab:
        render_experiments_tab(registry, selected_record)
    with logs_tab:
        render_logs_tab()


if __name__ == "__main__":
    main()
