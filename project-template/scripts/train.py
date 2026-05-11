"""
Baseline 학습 및 실험 기록 스크립트.

사용법:
    python scripts/train.py --config configs/default.yaml

데이터 파일이 없으면 scikit-learn 샘플 데이터로 실행 흐름을 확인합니다.
실제 프로젝트에서는 configs/default.yaml의 data.path와 target_column을 수정하세요.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def set_seed(seed: int = 42) -> None:
    """재현 가능한 학습을 위해 시드를 고정한다."""
    random.seed(seed)
    np.random.seed(seed)


def load_config(config_path: str) -> dict[str, Any]:
    """YAML 설정 파일을 불러온다."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def build_model(model_config: dict[str, Any], seed: int) -> Pipeline:
    """기본 baseline 모델을 생성한다."""
    model_name = model_config.get("name", "logistic_regression")
    params = dict(model_config.get("params", {}))

    if model_name != "logistic_regression":
        raise ValueError("현재 템플릿 baseline은 logistic_regression만 제공합니다.")

    params.setdefault("max_iter", 1000)
    params.setdefault("random_state", seed)
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(**params)),
        ]
    )


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """classification baseline metric을 계산한다."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_test, y_pred, average="macro")), 6),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


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


def write_data_manifest(path: Path, record: dict[str, Any]) -> None:
    """현재 학습에 사용한 데이터 버전을 기록한다."""
    manifest = read_json(path, {"datasets": []})
    datasets = manifest.setdefault("datasets", [])
    datasets.append(record)
    write_json(path, manifest)


def write_result_markdown(run_dir: Path, record: dict[str, Any], metrics: dict[str, Any]) -> None:
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
        "## Metrics",
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
            "## Notes",
            "",
            "- Baseline 실행 결과입니다. 실제 프로젝트에서는 가설, 변경점, 해석, 다음 액션을 추가하세요.",
            "- 자세한 비교는 `model_registry.json`과 `reports/EXPERIMENT_REPORT.md`에 정리하세요.",
            "",
        ]
    )
    (run_dir / "result.md").write_text("\n".join(lines), encoding="utf-8")


def write_predictions(run_dir: Path, y_true: pd.Series, y_pred: np.ndarray) -> None:
    """오류 분석을 위한 예측 샘플을 CSV로 저장한다."""
    with open(run_dir / "predictions.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["row_number", "y_true", "y_pred", "is_correct"])
        writer.writeheader()
        for idx, (actual, pred) in enumerate(zip(y_true.tolist(), y_pred.tolist())):
            writer.writerow(
                {
                    "row_number": idx,
                    "y_true": actual,
                    "y_pred": pred,
                    "is_correct": actual == pred,
                }
            )


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


def main(config_path: str) -> dict[str, Any]:
    config = load_config(config_path)
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

    model = build_model(config.get("model", {}), seed)
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    y_pred = model.predict(X_test)

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

    shutil.copy2(config_path, run_dir / "config.yaml")
    write_json(run_dir / "metrics.json", metrics)
    write_predictions(run_dir, y_test, y_pred)
    write_confusion_matrix(run_dir, y_test, y_pred)

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
        "artifact_path": str(artifact_path),
        "experiment_path": str(run_dir),
        "confusion_matrix_path": str(run_dir / "confusion_matrix.json"),
        "limitations": "Baseline only. 실제 프로젝트 데이터와 error analysis로 한계를 갱신하세요.",
    }

    write_json(run_dir / "data_manifest_entry.json", data_record)
    write_result_markdown(run_dir, model_record, metrics)
    write_data_manifest(Path(tracking_config.get("data_manifest_path", "data_manifest.json")), data_record)
    append_model_registry(Path(tracking_config.get("model_registry_path", "model_registry.json")), model_record)

    print(json.dumps({"run_id": run_id, "metrics": model_record["metrics"], "run_dir": str(run_dir)}, ensure_ascii=False, indent=2))
    return model_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
