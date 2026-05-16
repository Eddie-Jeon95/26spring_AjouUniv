"""
Baseline 학습 및 실험 기록 스크립트.

사용법:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --data data/processed/[processed_file] --target [target] --data-version [data_version]
    python scripts/train.py --data data/processed/[processed_file] --target [target] --data-version [data_version] --val-size 0.2 --test-size 0.2

데이터 파일이 없으면 scikit-learn 샘플 데이터로 실행 흐름을 확인합니다.
실제 프로젝트에서는 scripts/preprocess.py로 data/processed/ CSV를 만든 뒤 --data, --target, --data-version을 넘기세요.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CLASSIFICATION_METRICS = {
    "accuracy",
    "macro_f1",
    "precision_macro",
    "recall_macro",
    "roc_auc",
    "pr_auc",
}
REGRESSION_METRICS = {"mae", "rmse", "r2"}
DEFAULT_CLASSIFICATION_METRICS = ["accuracy", "macro_f1", "precision_macro", "recall_macro"]
DEFAULT_REGRESSION_METRICS = ["mae", "rmse", "r2"]


def set_seed(seed: int = 42) -> None:
    """재현 가능한 학습을 위해 시드를 고정한다."""
    random.seed(seed)
    np.random.seed(seed)


def load_config(config_path: str) -> dict[str, Any]:
    """YAML 설정 파일을 불러온다."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_csv_list(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """CLI/config의 comma separated metric list를 list로 변환한다."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def task_config(config: dict[str, Any]) -> dict[str, Any]:
    task = config.get("task", {})
    return task if isinstance(task, dict) else {}


def task_type_from_config(config: dict[str, Any]) -> str:
    task_type = task_config(config).get("type") or config.get("model", {}).get("task") or "classification"
    task_type = str(task_type).strip().lower()
    if task_type not in {"classification", "regression"}:
        raise ValueError("task.type은 classification 또는 regression이어야 합니다.")
    return task_type


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """학생이 YAML 대신 CLI로 넘긴 값을 effective config에 반영한다."""
    config = dict(config)
    data_config = dict(config.get("data", {}))
    split_config = dict(data_config.get("split", {}))
    tracking_config = dict(config.get("tracking", {}))
    task = dict(config.get("task", {}))

    if args.data:
        data_config["path"] = args.data
        data_config["source"] = args.data
        data_config["allow_demo_fallback"] = False
    if args.target:
        data_config["target_column"] = args.target
    if args.data_version:
        data_config["data_version"] = args.data_version
    if args.task_type:
        task["type"] = args.task_type
        model_config = dict(config.get("model", {}))
        model_config["task"] = args.task_type
        if args.task_type == "regression" and model_config.get("name") == "logistic_regression":
            model_config["name"] = "linear_regression"
            model_config["params"] = {}
        config["model"] = model_config
    if args.positive_class is not None:
        task["positive_class"] = args.positive_class
    if args.experiment_name:
        config["experiment_name"] = args.experiment_name
    if args.primary_metric:
        tracking_config["primary_metric"] = args.primary_metric
    if args.metrics:
        tracking_config["auxiliary_metrics"] = parse_csv_list(args.metrics)
    if args.model_name:
        model_config = dict(config.get("model", {}))
        if model_config.get("name") != args.model_name:
            model_config["params"] = {}  # 다른 모델의 기본 params 오염 방지
        model_config["name"] = args.model_name
        config["model"] = model_config
    if args.model_params:
        model_config = dict(config.get("model", {}))
        existing = dict(model_config.get("params", {}))
        for pair in args.model_params.split(","):
            pair = pair.strip()
            if "=" not in pair:
                continue
            k, v = pair.split("=", 1)
            k = k.strip()
            try:
                existing[k] = int(v)
            except ValueError:
                try:
                    existing[k] = float(v)
                except ValueError:
                    existing[k] = None if v.strip() == "None" else v.strip()
        model_config["params"] = existing
        config["model"] = model_config
    if args.test_size is not None:
        split_config["test_size"] = args.test_size
    if args.val_size is not None:
        split_config["val_size"] = args.val_size
    if args.no_stratify:
        split_config["stratify"] = False

    data_config["split"] = split_config
    config["data"] = data_config
    config["tracking"] = tracking_config
    config["task"] = task
    return config


def stable_hash(payload: Any) -> str:
    """dict/list 설정을 짧은 해시로 변환한다."""
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def file_sha256(path: Path) -> str:
    """파일 checksum을 계산한다."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(
    data_config: dict[str, Any],
    task_type: str = "classification",
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    """CSV 데이터가 있으면 사용하고, 없으면 demo 데이터를 반환한다."""
    data_path = Path(data_config.get("path", "data/raw/dataset.csv"))
    target_column = data_config.get("target_column", "label")
    allow_demo_fallback = bool(data_config.get("allow_demo_fallback", True))

    if data_path.exists():
        df = pd.read_csv(data_path)
        if target_column not in df.columns:
            raise ValueError(f"target_column '{target_column}'이 {data_path}에 없습니다.")
        source = str(data_path)
        metadata = {
            "source": data_config.get("source", source),
            "file_path": source,
            "checksum_sha256": file_sha256(data_path),
            "fallback_demo_data": False,
        }
        return df, target_column, metadata

    if not allow_demo_fallback:
        raise FileNotFoundError(
            f"학습 데이터 파일을 찾을 수 없습니다: {data_path}\n"
            "먼저 scripts/preprocess.py로 data/processed/ CSV를 만들거나 --data 경로를 확인하세요."
        )

    if task_type == "regression":
        demo = load_diabetes(as_frame=True)
        df = demo.frame.rename(columns={"target": target_column})
        source = "sklearn.datasets.load_diabetes"
    else:
        demo = load_breast_cancer(as_frame=True)
        df = demo.frame.rename(columns={"target": target_column})
        source = "sklearn.datasets.load_breast_cancer"

    metadata = {
        "source": source,
        "file_path": None,
        "checksum_sha256": stable_hash(df.head(50).to_dict(orient="list")),
        "fallback_demo_data": True,
    }
    return df, target_column, metadata


def split_dataset(
    df: pd.DataFrame,
    target_column: str,
    seed: int,
    split_config: dict[str, Any],
    task_type: str = "classification",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """train/validation/test 분리를 수행한다."""
    test_size = float(split_config.get("test_size", 0.2))
    val_size = float(split_config.get("val_size", 0.1))
    stratify_enabled = bool(split_config.get("stratify", True)) and task_type == "classification"
    if not 0 < test_size < 1:
        raise ValueError("split.test_size는 0과 1 사이여야 합니다.")
    if not 0 <= val_size < 1:
        raise ValueError("split.val_size는 0 이상 1 미만이어야 합니다.")
    if test_size + val_size >= 1:
        raise ValueError("split.test_size + split.val_size는 1보다 작아야 합니다.")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    stratify = y if stratify_enabled and y.nunique() > 1 else None

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    relative_val_size = val_size / (1.0 - test_size)
    stratify_train_val = y_train_val if stratify_enabled and y_train_val.nunique() > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val_size,
        random_state=seed,
        stratify=stratify_train_val,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def split_feature_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """mixed tabular baseline을 위한 numeric/categorical 컬럼을 나눈다."""
    categorical_columns = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_columns = [
        column
        for column in X.columns
        if column not in categorical_columns and pd.api.types.is_numeric_dtype(X[column])
    ]
    categorical_columns.extend(
        column for column in X.columns if column not in numeric_columns and column not in categorical_columns
    )
    return numeric_columns, categorical_columns


_SUPPORTED_MODELS = ("logistic_regression", "random_forest", "linear_regression", "random_forest_regressor")


def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    use_scaler: bool,
) -> ColumnTransformer:
    """numeric/categorical 컬럼에 맞는 ColumnTransformer를 반환한다."""
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if use_scaler:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps=numeric_steps), numeric_columns))
    if categorical_columns:
        transformers.append((
            "categorical",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical_columns,
        ))
    if not transformers:
        raise ValueError("학습에 사용할 feature column이 없습니다.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_model(model_config: dict[str, Any], seed: int, X_train: pd.DataFrame, task_type: str) -> Pipeline:
    """tabular baseline 모델 Pipeline을 생성한다."""
    default_model_name = "linear_regression" if task_type == "regression" else "logistic_regression"
    model_name = model_config.get("name", default_model_name)
    params = dict(model_config.get("params", {}))
    if task_type == "regression" and model_name == "logistic_regression":
        model_name = "linear_regression"
        params = {}

    if model_name not in _SUPPORTED_MODELS:
        raise ValueError(
            f"지원하지 않는 모델: '{model_name}'. 지원 목록: {_SUPPORTED_MODELS}"
        )

    numeric_columns, categorical_columns = split_feature_columns(X_train)

    if task_type == "classification" and model_name == "logistic_regression":
        params.setdefault("max_iter", 1000)
        params.setdefault("random_state", seed)
        estimator: Any = LogisticRegression(**params)
        use_scaler = True
    elif task_type == "classification" and model_name == "random_forest":
        params.setdefault("random_state", seed)
        if params.get("class_weight") == "None":
            params["class_weight"] = None
        estimator = RandomForestClassifier(**params)
        use_scaler = False
    elif task_type == "regression" and model_name == "linear_regression":
        estimator = LinearRegression(**params)
        use_scaler = True
    elif task_type == "regression" and model_name in {"random_forest", "random_forest_regressor"}:
        params.setdefault("random_state", seed)
        estimator = RandomForestRegressor(**params)
        use_scaler = False
    else:
        raise ValueError(f"'{model_name}'은 {task_type} task에서 사용할 수 없습니다.")

    preprocessor = build_preprocessor(numeric_columns, categorical_columns, use_scaler)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])


def jsonable(value: Any) -> Any:
    """numpy/pandas scalar를 JSON 직렬화 가능한 값으로 변환한다."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value) if not isinstance(value, (list, dict, tuple, np.ndarray)) else False:
        return None
    return value


def metric_names_for_task(
    task_type: str,
    primary_metric: str | None,
    auxiliary_metrics: list[str] | None,
) -> list[str]:
    defaults = DEFAULT_REGRESSION_METRICS if task_type == "regression" else DEFAULT_CLASSIFICATION_METRICS
    names = []
    if primary_metric:
        names.append(primary_metric)
    names.extend(auxiliary_metrics or [])
    names.extend(defaults)
    allowed = REGRESSION_METRICS if task_type == "regression" else CLASSIFICATION_METRICS
    ordered: list[str] = []
    for name in names:
        metric = str(name).strip()
        if not metric:
            continue
        if metric not in allowed:
            raise ValueError(f"{task_type}에서 지원하지 않는 metric입니다: {metric}")
        if metric not in ordered:
            ordered.append(metric)
    return ordered


def infer_positive_label(y: pd.Series, positive_class: Any | None = None) -> Any | None:
    """binary classification의 positive class를 확정한다."""
    classes = list(pd.Series(y).drop_duplicates())
    if len(classes) != 2:
        return None
    if positive_class is None or positive_class == "":
        return sorted(classes, key=lambda value: str(value))[-1]
    for value in classes:
        if str(value) == str(positive_class):
            return value
    raise ValueError(f"positive_class '{positive_class}'가 target 값 {classes}에 없습니다.")


def positive_class_score(model: Any, X_eval: pd.DataFrame, positive_label: Any | None) -> np.ndarray | None:
    """predict_proba가 가능하면 positive class probability를 반환한다."""
    if positive_label is None or not hasattr(model, "predict_proba"):
        return None
    try:
        classes = list(getattr(model, "classes_", []))
        if positive_label not in classes:
            return None
        proba = model.predict_proba(X_eval)
        return np.asarray(proba)[:, classes.index(positive_label)]
    except Exception:  # noqa: BLE001
        return None


def evaluate_classification(
    model: Any,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    metric_names: list[str],
    positive_label: Any | None,
) -> dict[str, Any]:
    """classification metric을 계산한다."""
    y_pred = model.predict(X_eval)
    metrics: dict[str, Any] = {}
    if "accuracy" in metric_names:
        metrics["accuracy"] = round(float(accuracy_score(y_eval, y_pred)), 6)
    if "macro_f1" in metric_names:
        metrics["macro_f1"] = round(float(f1_score(y_eval, y_pred, average="macro", zero_division=0)), 6)
    if "precision_macro" in metric_names:
        metrics["precision_macro"] = round(
            float(precision_score(y_eval, y_pred, average="macro", zero_division=0)), 6
        )
    if "recall_macro" in metric_names:
        metrics["recall_macro"] = round(float(recall_score(y_eval, y_pred, average="macro", zero_division=0)), 6)

    y_score = positive_class_score(model, X_eval, positive_label)
    if y_score is not None:
        y_true_binary = (pd.Series(y_eval).reset_index(drop=True) == positive_label).astype(int)
        if "roc_auc" in metric_names and y_true_binary.nunique() == 2:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true_binary, y_score)), 6)
        if "pr_auc" in metric_names and y_true_binary.nunique() == 2:
            metrics["pr_auc"] = round(float(average_precision_score(y_true_binary, y_score)), 6)

    metrics["classification_report"] = classification_report(y_eval, y_pred, output_dict=True, zero_division=0)
    return metrics


def evaluate_regression(
    model: Any,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    metric_names: list[str],
) -> dict[str, Any]:
    """regression metric을 계산한다."""
    y_pred = model.predict(X_eval)
    metrics: dict[str, Any] = {}
    if "mae" in metric_names:
        metrics["mae"] = round(float(mean_absolute_error(y_eval, y_pred)), 6)
    if "rmse" in metric_names:
        metrics["rmse"] = round(float(root_mean_squared_error(y_eval, y_pred)), 6)
    if "r2" in metric_names:
        metrics["r2"] = round(float(r2_score(y_eval, y_pred)), 6)
    return metrics


def evaluate(
    model: Any,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    task_type: str,
    metric_names: list[str],
    positive_label: Any | None = None,
) -> dict[str, Any]:
    """task type에 맞는 metric을 계산한다."""
    if task_type == "regression":
        return evaluate_regression(model, X_eval, y_eval, metric_names)
    return evaluate_classification(model, X_eval, y_eval, metric_names, positive_label)


def validate_primary_metric(metrics: dict[str, Any], primary_metric: str) -> None:
    if primary_metric not in metrics:
        available = ", ".join(k for k in metrics if k != "classification_report")
        raise ValueError(
            f"primary_metric '{primary_metric}'이 계산되지 않았습니다. 사용 가능한 metric: {available}"
        )


def threshold_rows(
    y_true: pd.Series,
    y_score: np.ndarray | None,
    positive_label: Any | None,
    negative_label: Any | None,
) -> list[dict[str, Any]]:
    """threshold별 precision/recall/f1/confusion counts를 계산한다."""
    if y_score is None or positive_label is None or negative_label is None:
        return []
    y_true_binary = (pd.Series(y_true).reset_index(drop=True) == positive_label).astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    for threshold in np.linspace(0, 1, 101):
        pred_binary = (y_score >= threshold).astype(int)
        tp = int(((pred_binary == 1) & (y_true_binary == 1)).sum())
        fp = int(((pred_binary == 1) & (y_true_binary == 0)).sum())
        tn = int(((pred_binary == 0) & (y_true_binary == 0)).sum())
        fn = int(((pred_binary == 0) & (y_true_binary == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({
            "threshold": round(float(threshold), 2),
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "f1": round(float(f1), 6),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "positive_class": str(positive_label),
            "negative_class": str(negative_label),
        })
    return rows


def write_threshold_metrics(run_dir: Path, rows: list[dict[str, Any]]) -> Path | None:
    if not rows:
        return None
    path = run_dir / "threshold_metrics.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(jsonable(payload), f, allow_unicode=True, sort_keys=False)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def append_model_registry(path: Path, record: dict[str, Any]) -> None:
    """model_registry.json에 실험 결과를 추가한다."""
    registry = read_json(path, [])
    if not isinstance(registry, list):
        raise ValueError(f"{path}는 list 형태의 JSON이어야 합니다.")
    registry.append(record)
    write_json(path, registry)


def write_result_markdown(
    run_dir: Path,
    record: dict[str, Any],
    metrics: dict[str, Any],
    test_metrics: dict[str, Any],
) -> None:
    """실험 요약 Markdown을 저장한다."""
    lines = [
        f"# Experiment: {record['model_id']}",
        "",
        f"- Run ID: `{record['run_id']}`",
        f"- Data version: `{record['data_version']}`",
        f"- Task type: `{record.get('task_type', 'classification')}`",
        f"- Model: `{record['model_name']}`",
        f"- Config hash: `{record['config_hash']}`",
        f"- Artifact: `{record['artifact_path']}`",
    ]
    if record.get("confusion_matrix_path"):
        lines.append(f"- Confusion matrix: `{record['confusion_matrix_path']}`")
    if record.get("threshold_metrics_path"):
        lines.append(f"- Threshold metrics: `{record['threshold_metrics_path']}`")
    lines.extend([
        "",
        "## Validation Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ])
    for metric_name, value in metrics.items():
        if metric_name != "classification_report":
            lines.append(f"| {metric_name} | {value} |")
    lines.extend(
        [
            "",
            "## Test Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
    )
    for metric_name, value in test_metrics.items():
        if metric_name != "classification_report":
            lines.append(f"| {metric_name} | {value} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Validation metric은 실험 비교 기준입니다. Test metric은 최종 확인용으로 해석하세요.",
            "- Threshold metric은 validation split 기준이며, test set으로 threshold를 선택하지 않습니다.",
            "- 자세한 비교는 `model_registry.json`과 `reports/EXPERIMENT_REPORT.md`에 정리하세요.",
            "",
        ]
    )
    (run_dir / "result.md").write_text("\n".join(lines), encoding="utf-8")


def write_predictions(
    run_dir: Path,
    X_eval: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    prediction_probability: np.ndarray | None = None,
    positive_probability: np.ndarray | None = None,
) -> None:
    """오류 분석을 위한 예측 샘플과 입력 feature를 CSV로 저장한다."""
    fieldnames = [
        "row_number",
        "original_index",
        *[str(column) for column in X_eval.columns],
        "y_true",
        "y_pred",
        "is_correct",
    ]
    if prediction_probability is not None:
        fieldnames.append("prediction_probability")
    if positive_probability is not None:
        fieldnames.append("positive_probability")

    with open(run_dir / "predictions.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row_number, (original_index, features) in enumerate(X_eval.iterrows()):
            actual = y_true.iloc[row_number]
            pred = y_pred[row_number]
            row = {
                "row_number": row_number,
                "original_index": original_index,
                **{str(column): features[column] for column in X_eval.columns},
                "y_true": actual,
                "y_pred": pred,
                "is_correct": actual == pred,
            }
            if prediction_probability is not None:
                row["prediction_probability"] = round(float(prediction_probability[row_number]), 6)
            if positive_probability is not None:
                row["positive_probability"] = round(float(positive_probability[row_number]), 6)
            writer.writerow(row)


def write_confusion_matrix(run_dir: Path, y_true: pd.Series, y_pred: np.ndarray) -> Path:
    """Streamlit 대시보드에서 사용할 confusion matrix를 JSON으로 저장한다."""
    labels = pd.Index(
        pd.concat([y_true.reset_index(drop=True), pd.Series(y_pred)]).drop_duplicates()
    ).tolist()
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    payload = {
        "labels": [str(label) for label in labels],
        "matrix": matrix.astype(int).tolist(),
        "format": "rows=true_label, columns=predicted_label",
    }
    path = run_dir / "confusion_matrix.json"
    write_json(path, payload)
    return path


def class_labels(model: Any, y: pd.Series) -> list[Any]:
    labels = list(getattr(model, "classes_", []))
    return labels if labels else sorted(pd.Series(y).drop_duplicates().tolist(), key=lambda value: str(value))


def main(config_path: str, args: argparse.Namespace | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    if args is not None:
        config = apply_cli_overrides(config, args)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    task_type = task_type_from_config(config)
    tracking_config = config.get("tracking", {})
    primary_metric = tracking_config.get("primary_metric") or ("rmse" if task_type == "regression" else "accuracy")
    auxiliary_metrics = parse_csv_list(tracking_config.get("auxiliary_metrics"))
    metric_names = metric_names_for_task(task_type, primary_metric, auxiliary_metrics)

    df, target_column, data_metadata = load_dataset(config.get("data", {}), task_type)
    split_config = config.get("data", {}).get("split", {})
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        df=df,
        target_column=target_column,
        seed=seed,
        split_config=split_config,
        task_type=task_type,
    )

    positive_label = infer_positive_label(y_train, task_config(config).get("positive_class")) if task_type == "classification" else None
    model = build_model(config.get("model", {}), seed, X_train, task_type)
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_val, y_val, task_type, metric_names, positive_label)
    test_metrics = evaluate(model, X_test, y_test, task_type, metric_names, positive_label)
    validate_primary_metric(metrics, primary_metric)

    y_pred = model.predict(X_val)
    prediction_probability = None
    positive_probability = positive_class_score(model, X_val, positive_label)
    if task_type == "classification" and hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_val)
        labels = class_labels(model, y_train)
        predicted_class_positions = [labels.index(pred) for pred in y_pred]
        prediction_probability = np.array(
            [y_proba[row_index, class_index] for row_index, class_index in enumerate(predicted_class_positions)]
        )

    now = datetime.now()
    config_hash = stable_hash(config)
    experiment_name = config.get("experiment_name", config.get("model", {}).get("name", "baseline"))
    run_id = f"{now:%Y%m%d_%H%M%S}_{experiment_name}_{config_hash}"

    runs_dir = Path(tracking_config.get("runs_dir", "experiments/runs"))
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model_save_dir = Path(config.get("model", {}).get("save_dir", "models"))
    model_save_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = model_save_dir / f"{run_id}.pkl"
    with open(artifact_path, "wb") as f:
        pickle.dump(model, f)

    write_yaml(run_dir / "config.yaml", config)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "test_metrics.json", test_metrics)
    write_predictions(run_dir, X_val, y_val, y_pred, prediction_probability, positive_probability)

    confusion_matrix_path = None
    threshold_metrics_path = None
    if task_type == "classification":
        confusion_matrix_path = write_confusion_matrix(run_dir, y_val, y_pred)
        labels = class_labels(model, y_train)
        negative_label = next((label for label in labels if label != positive_label), None)
        threshold_metrics_path = write_threshold_metrics(
            run_dir,
            threshold_rows(y_val, positive_probability, positive_label, negative_label),
        )

    data_version = config.get("data", {}).get("data_version") or f"data-{config_hash}"
    data_record = {
        "data_version": data_version,
        "created_at": now.isoformat(timespec="seconds"),
        "target_column": target_column,
        "task_type": task_type,
        "positive_class": str(positive_label) if positive_label is not None else None,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "split_seed": seed,
        "split": {
            "train_rows": int(len(X_train)),
            "val_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
            **split_config,
        },
        **data_metadata,
    }

    metric_payload = {k: v for k, v in metrics.items() if k != "classification_report"}
    test_metric_payload = {k: v for k, v in test_metrics.items() if k != "classification_report"}
    model_name = model.named_steps["model"].__class__.__name__
    model_record = {
        "model_id": f"{model_name.lower()}-{config_hash}",
        "run_id": run_id,
        "created_at": now.isoformat(timespec="seconds"),
        "experiment_type": "baseline",
        "model_name": model_name,
        "task_type": task_type,
        "positive_class": str(positive_label) if positive_label is not None else None,
        "data_version": data_version,
        "config_hash": config_hash,
        "primary_metric": primary_metric,
        "auxiliary_metrics": [metric for metric in metric_names if metric != primary_metric],
        "metrics": metric_payload,
        "test_metrics": test_metric_payload,
        "artifact_path": str(artifact_path),
        "experiment_path": str(run_dir),
        "confusion_matrix_path": str(confusion_matrix_path) if confusion_matrix_path else None,
        "threshold_metrics_path": str(threshold_metrics_path) if threshold_metrics_path else None,
        "limitations": "Baseline only. Validation metrics are for experiment comparison; test metrics are final-check context.",
    }

    write_json(run_dir / "data_manifest_entry.json", data_record)
    write_result_markdown(run_dir, model_record, metrics, test_metrics)
    append_model_registry(Path(tracking_config.get("model_registry_path", "model_registry.json")), model_record)

    print(
        json.dumps(
            {
                "run_id": run_id,
                "metrics": model_record["metrics"],
                "test_metrics": model_record["test_metrics"],
                "run_dir": str(run_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return model_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data", type=str, default=None, help="학습용 processed CSV 경로")
    parser.add_argument("--target", type=str, default=None, help="target column 이름")
    parser.add_argument("--data-version", type=str, default=None, help="실험에 사용할 data_version")
    parser.add_argument("--task-type", choices=["classification", "regression"], default=None, help="task 종류")
    parser.add_argument("--positive-class", type=str, default=None, help="binary classification positive class")
    parser.add_argument("--experiment-name", type=str, default=None, help="run_id에 들어갈 실험 이름")
    parser.add_argument("--primary-metric", type=str, default=None, help="기록할 primary metric 이름")
    parser.add_argument("--metrics", type=str, default=None, help="보조 metric 목록. 예: accuracy,macro_f1,roc_auc")
    parser.add_argument("--test-size", type=float, default=None, help="전체 데이터 중 test 비율")
    parser.add_argument("--val-size", type=float, default=None, help="전체 데이터 중 validation 비율")
    parser.add_argument("--no-stratify", action="store_true", help="stratified split을 끕니다")
    parser.add_argument("--model-name", type=str, default=None,
                        help=f"사용할 모델 이름. 지원: {_SUPPORTED_MODELS}")
    parser.add_argument("--model-params", type=str, default=None,
                        help="모델 하이퍼파라미터. 예: n_estimators=100,max_depth=5")
    args = parser.parse_args()
    try:
        main(args.config, args)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
