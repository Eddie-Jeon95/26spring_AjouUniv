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
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def set_seed(seed: int = 42) -> None:
    """재현 가능한 학습을 위해 시드를 고정한다."""
    random.seed(seed)
    np.random.seed(seed)


def load_config(config_path: str) -> dict[str, Any]:
    """YAML 설정 파일을 불러온다."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """학생이 YAML 대신 CLI로 넘긴 값을 effective config에 반영한다."""
    config = dict(config)
    data_config = dict(config.get("data", {}))
    split_config = dict(data_config.get("split", {}))
    tracking_config = dict(config.get("tracking", {}))

    if args.data:
        data_config["path"] = args.data
        data_config["source"] = args.data
        data_config["allow_demo_fallback"] = False
    if args.target:
        data_config["target_column"] = args.target
    if args.data_version:
        data_config["data_version"] = args.data_version
    if args.experiment_name:
        config["experiment_name"] = args.experiment_name
    if args.primary_metric:
        tracking_config["primary_metric"] = args.primary_metric
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


def load_dataset(data_config: dict[str, Any]) -> tuple[pd.DataFrame, str, dict[str, Any]]:
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

    demo = load_breast_cancer(as_frame=True)
    df = demo.frame.rename(columns={"target": target_column})
    metadata = {
        "source": "sklearn.datasets.load_breast_cancer",
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """train/validation/test 분리를 수행한다."""
    test_size = float(split_config.get("test_size", 0.2))
    val_size = float(split_config.get("val_size", 0.1))
    stratify_enabled = bool(split_config.get("stratify", True))
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


_SUPPORTED_MODELS = ("logistic_regression", "random_forest")


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


def build_model(model_config: dict[str, Any], seed: int, X_train: pd.DataFrame) -> Pipeline:
    """tabular classification 모델 Pipeline을 생성한다.

    지원 모델: logistic_regression, random_forest
    """
    model_name = model_config.get("name", "logistic_regression")
    params = dict(model_config.get("params", {}))

    if model_name not in _SUPPORTED_MODELS:
        raise ValueError(
            f"지원하지 않는 모델: '{model_name}'. 지원 목록: {_SUPPORTED_MODELS}"
        )

    numeric_columns, categorical_columns = split_feature_columns(X_train)

    if model_name == "logistic_regression":
        params.setdefault("max_iter", 1000)
        params.setdefault("random_state", seed)
        estimator: Any = LogisticRegression(**params)
        use_scaler = True

    elif model_name == "random_forest":
        params.setdefault("random_state", seed)
        # class_weight 값이 문자열 None이면 파이썬 None으로 변환
        if params.get("class_weight") == "None":
            params["class_weight"] = None
        estimator = RandomForestClassifier(**params)
        use_scaler = False  # 트리 계열은 feature scale에 무관

    preprocessor = build_preprocessor(numeric_columns, categorical_columns, use_scaler)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])


def evaluate(model: Pipeline, X_eval: pd.DataFrame, y_eval: pd.Series) -> dict[str, Any]:
    """classification baseline metric을 계산한다."""
    y_pred = model.predict(X_eval)
    return {
        "accuracy": round(float(accuracy_score(y_eval, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_eval, y_pred, average="macro")), 6),
        "classification_report": classification_report(y_eval, y_pred, output_dict=True),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


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
        f"- Model: `{record['model_name']}`",
        f"- Config hash: `{record['config_hash']}`",
        f"- Artifact: `{record['artifact_path']}`",
        f"- Confusion matrix: `{record['confusion_matrix_path']}`",
        "",
        "## Validation Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for metric_name in ("accuracy", "macro_f1"):
        if metric_name in metrics:
            lines.append(f"| {metric_name} | {metrics[metric_name]} |")
    lines.extend(
        [
            "",
            "## Test Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
    )
    for metric_name in ("accuracy", "macro_f1"):
        if metric_name in test_metrics:
            lines.append(f"| {metric_name} | {test_metrics[metric_name]} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Validation metric은 실험 비교 기준입니다. Test metric은 최종 확인용으로 해석하세요.",
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
            writer.writerow(row)


def write_confusion_matrix(run_dir: Path, y_true: pd.Series, y_pred: np.ndarray) -> None:
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
    write_json(run_dir / "confusion_matrix.json", payload)


def main(config_path: str, args: argparse.Namespace | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    if args is not None:
        config = apply_cli_overrides(config, args)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    df, target_column, data_metadata = load_dataset(config.get("data", {}))
    split_config = config.get("data", {}).get("split", {})
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        df=df,
        target_column=target_column,
        seed=seed,
        split_config=split_config,
    )

    model = build_model(config.get("model", {}), seed, X_train)
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_val, y_val)
    test_metrics = evaluate(model, X_test, y_test)
    y_pred = model.predict(X_val)
    prediction_probability = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_val)
        predicted_class_positions = [list(model.classes_).index(pred) for pred in y_pred]
        prediction_probability = np.array(
            [y_proba[row_index, class_index] for row_index, class_index in enumerate(predicted_class_positions)]
        )

    now = datetime.now()
    config_hash = stable_hash(config)
    experiment_name = config.get("experiment_name", config.get("model", {}).get("name", "baseline"))
    run_id = f"{now:%Y%m%d_%H%M%S}_{experiment_name}_{config_hash}"

    tracking_config = config.get("tracking", {})
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
    write_predictions(run_dir, X_val, y_val, y_pred, prediction_probability)
    write_confusion_matrix(run_dir, y_val, y_pred)

    data_version = config.get("data", {}).get("data_version") or f"data-{config_hash}"
    data_record = {
        "data_version": data_version,
        "created_at": now.isoformat(timespec="seconds"),
        "target_column": target_column,
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

    primary_metric = tracking_config.get("primary_metric", "accuracy")
    model_record = {
        "model_id": f"{config.get('model', {}).get('name', 'model')}-{config_hash}",
        "run_id": run_id,
        "created_at": now.isoformat(timespec="seconds"),
        "model_name": config.get("model", {}).get("name", "logistic_regression"),
        "data_version": data_version,
        "config_hash": config_hash,
        "primary_metric": primary_metric,
        "metrics": {k: v for k, v in metrics.items() if k != "classification_report"},
        "test_metrics": {k: v for k, v in test_metrics.items() if k != "classification_report"},
        "artifact_path": str(artifact_path),
        "experiment_path": str(run_dir),
        "confusion_matrix_path": str(run_dir / "confusion_matrix.json"),
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
    parser.add_argument("--experiment-name", type=str, default=None, help="run_id에 들어갈 실험 이름")
    parser.add_argument("--primary-metric", type=str, default=None, help="기록할 primary metric 이름")
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
