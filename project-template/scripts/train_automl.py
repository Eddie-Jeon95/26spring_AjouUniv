"""
AutoGluon Tabular 기반 AutoML 학습 및 실험 기록 스크립트.

AutoML은 baseline 이후 같은 data_version/split/metric 조건에서 pipeline 후보를 비교하기 위해 사용합니다.
test set은 AutoGluon 탐색에 전달하지 않고 최종 확인과 leaderboard 표시용으로만 사용합니다.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from decision_blocks import load_decision_block
from train import (
    append_model_registry,
    apply_training_decisions,
    evaluate,
    infer_positive_label,
    load_config,
    load_dataset,
    metric_names_for_task,
    parse_csv_list,
    positive_class_score,
    read_json,
    set_seed,
    split_dataset,
    stable_hash,
    threshold_rows,
    validate_training_decisions,
    validate_primary_metric,
    write_confusion_matrix,
    write_json,
    write_predictions,
    write_result_markdown,
    write_threshold_metrics,
    write_yaml,
)


AUTOGLUON_METRIC_MAP = {
    "accuracy": "accuracy",
    "macro_f1": "f1_macro",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "mae": "mean_absolute_error",
    "rmse": "root_mean_squared_error",
    "r2": "r2",
}


class AutoGluonAdapter:
    """AutoGluon predictor를 sklearn-like 평가 helper에 맞춘 얇은 adapter."""

    def __init__(self, predictor: Any, classes: list[Any]) -> None:
        self.predictor = predictor
        self.classes_ = classes

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.predictor.predict(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        try:
            proba = self.predictor.predict_proba(X, as_multiclass=True)
        except TypeError:
            proba = self.predictor.predict_proba(X)

        if isinstance(proba, pd.DataFrame):
            columns = list(proba.columns)
            ordered = [label for label in self.classes_ if label in columns]
            if len(ordered) == len(self.classes_):
                return proba[ordered].to_numpy()
            return proba.to_numpy()

        if isinstance(proba, pd.Series):
            positive = proba.to_numpy()
            return np.column_stack([1 - positive, positive])

        array = np.asarray(proba)
        if array.ndim == 1:
            return np.column_stack([1 - array, array])
        return array


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = dict(config)
    data_config = dict(config.get("data", {}))
    split_config = dict(data_config.get("split", {}))
    task = dict(config.get("task", {}))
    tracking = dict(config.get("tracking", {}))
    automl = dict(config.get("automl", {}))

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
    if args.positive_class is not None:
        task["positive_class"] = args.positive_class
    if args.primary_metric:
        tracking["primary_metric"] = args.primary_metric
    if args.metrics:
        tracking["auxiliary_metrics"] = parse_csv_list(args.metrics)
    if args.test_size is not None:
        split_config["test_size"] = args.test_size
    if args.val_size is not None:
        split_config["val_size"] = args.val_size
    if args.no_stratify:
        split_config["stratify"] = False
    if args.time_limit is not None:
        automl["time_limit"] = args.time_limit
    if args.presets:
        automl["presets"] = args.presets

    data_config["split"] = split_config
    config["data"] = data_config
    config["task"] = task
    config["tracking"] = tracking
    config["automl"] = automl
    return config


def task_type_from_config(config: dict[str, Any]) -> str:
    task_type = str(config.get("task", {}).get("type") or "classification").lower()
    if task_type not in {"classification", "regression"}:
        raise ValueError("task.type은 classification 또는 regression이어야 합니다.")
    return task_type


def problem_type(task_type: str, y: pd.Series) -> str:
    if task_type == "regression":
        return "regression"
    return "binary" if y.nunique() == 2 else "multiclass"


def autogluon_metric_name(primary_metric: str) -> str:
    if primary_metric not in AUTOGLUON_METRIC_MAP:
        raise ValueError(f"AutoGluon metric으로 매핑할 수 없습니다: {primary_metric}")
    return AUTOGLUON_METRIC_MAP[primary_metric]


def parse_presets(value: Any) -> str | list[str]:
    if value is None:
        return "medium_quality"
    presets = parse_csv_list(value)
    if len(presets) > 1:
        return presets
    return presets[0] if presets else "medium_quality"


def selected_model_name(predictor: Any) -> str:
    model_best = getattr(predictor, "model_best", None)
    if model_best:
        return str(model_best)
    try:
        return str(predictor.get_model_best())
    except Exception:  # noqa: BLE001
        return "autogluon_best"


def write_leaderboard(predictor: Any, test_df: pd.DataFrame, run_dir: Path) -> Path:
    try:
        leaderboard = predictor.leaderboard(test_df, silent=True)
    except TypeError:
        leaderboard = predictor.leaderboard(test_df)
    path = run_dir / "leaderboard.csv"
    leaderboard.to_csv(path, index=False)
    return path


def main(config_path: str, args: argparse.Namespace | None = None) -> dict[str, Any]:
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise ImportError(
            "AutoGluon이 설치되어 있지 않습니다. AutoML 실행 전 `pip install -r requirements.txt`를 실행하세요."
        ) from exc

    config = load_config(config_path)
    if args is not None and getattr(args, "decisions", None):
        decisions = load_decision_block(args.decisions, "training_decisions")
        validate_training_decisions(decisions, args)
        config = apply_training_decisions(config, decisions)
    if args is not None:
        config = apply_cli_overrides(config, args)

    seed = int(config.get("seed", 42))
    set_seed(seed)
    task_type = task_type_from_config(config)
    tracking_config = config.get("tracking", {})
    primary_metric = tracking_config.get("primary_metric") or ("rmse" if task_type == "regression" else "accuracy")
    auxiliary_metrics = parse_csv_list(tracking_config.get("auxiliary_metrics"))
    metric_names = metric_names_for_task(task_type, primary_metric, auxiliary_metrics)
    eval_metric = autogluon_metric_name(primary_metric)

    df, target_column, data_metadata = load_dataset(config.get("data", {}), task_type)
    split_config = config.get("data", {}).get("split", {})
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        df=df,
        target_column=target_column,
        seed=seed,
        split_config=split_config,
        task_type=task_type,
    )
    train_df = X_train.copy()
    train_df[target_column] = y_train.to_numpy()
    val_df = X_val.copy()
    val_df[target_column] = y_val.to_numpy()
    test_df = X_test.copy()
    test_df[target_column] = y_test.to_numpy()

    positive_label = infer_positive_label(y_train, config.get("task", {}).get("positive_class")) if task_type == "classification" else None
    ag_problem_type = problem_type(task_type, y_train)

    now = datetime.now()
    config_hash = stable_hash(config)
    run_id = f"{now:%Y%m%d_%H%M%S}_autogluon_{config_hash}"
    tracking_config = config.get("tracking", {})
    runs_dir = Path(tracking_config.get("runs_dir", "experiments/runs"))
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    artifact_dir = Path(config.get("automl", {}).get("save_dir", "models/automl")) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    time_limit = int(config.get("automl", {}).get("time_limit", 300))
    presets = parse_presets(config.get("automl", {}).get("presets", "medium_quality"))

    predictor_kwargs = {
        "label": target_column,
        "problem_type": ag_problem_type,
        "eval_metric": eval_metric,
        "path": str(artifact_dir),
        "verbosity": 2,
    }
    if positive_label is not None and ag_problem_type == "binary":
        predictor_kwargs["positive_class"] = positive_label

    predictor = TabularPredictor(**predictor_kwargs).fit(
        train_data=train_df,
        tuning_data=val_df,
        time_limit=time_limit,
        presets=presets,
    )

    classes = list(getattr(predictor, "class_labels", []) or sorted(y_train.drop_duplicates().tolist(), key=lambda value: str(value)))
    adapter = AutoGluonAdapter(predictor, classes)
    metrics = evaluate(adapter, X_val, y_val, task_type, metric_names, positive_label)
    test_metrics = evaluate(adapter, X_test, y_test, task_type, metric_names, positive_label)
    validate_primary_metric(metrics, primary_metric)

    y_pred = adapter.predict(X_val)
    prediction_probability = None
    positive_probability = positive_class_score(adapter, X_val, positive_label)
    if task_type == "classification":
        y_proba = adapter.predict_proba(X_val)
        predicted_class_positions = [classes.index(pred) for pred in y_pred]
        prediction_probability = np.array(
            [y_proba[row_index, class_index] for row_index, class_index in enumerate(predicted_class_positions)]
        )

    leaderboard_path = write_leaderboard(predictor, test_df, run_dir)
    write_yaml(run_dir / "config.yaml", config)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "test_metrics.json", test_metrics)
    write_predictions(run_dir, X_val, y_val, y_pred, prediction_probability, positive_probability)

    confusion_matrix_path = None
    threshold_metrics_path = None
    if task_type == "classification":
        confusion_matrix_path = write_confusion_matrix(run_dir, y_val, y_pred)
        negative_label = next((label for label in classes if label != positive_label), None)
        threshold_metrics_path = write_threshold_metrics(
            run_dir,
            threshold_rows(y_val, positive_probability, positive_label, negative_label),
        )

    data_version = config.get("data", {}).get("data_version") or f"data-{config_hash}"
    selected_model = selected_model_name(predictor)
    automl_summary = {
        "backend": "autogluon",
        "problem_type": ag_problem_type,
        "task_type": task_type,
        "eval_metric": eval_metric,
        "primary_metric": primary_metric,
        "selected_model": selected_model,
        "positive_class": str(positive_label) if positive_label is not None else None,
        "time_limit": time_limit,
        "presets": presets,
        "data_version": data_version,
        "split": {
            "train_rows": int(len(X_train)),
            "val_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
            **split_config,
        },
        "leakage_constraints": "Use the same processed data, target, split, and excluded leakage columns as baseline. Test set is final-check only.",
    }
    automl_summary_path = run_dir / "automl_summary.json"
    write_json(automl_summary_path, automl_summary)

    data_record = {
        "data_version": data_version,
        "created_at": now.isoformat(timespec="seconds"),
        "target_column": target_column,
        "task_type": task_type,
        "positive_class": str(positive_label) if positive_label is not None else None,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "split_seed": seed,
        "split": automl_summary["split"],
        **data_metadata,
    }

    model_record = {
        "model_id": f"autogluon-{selected_model}-{config_hash}",
        "run_id": run_id,
        "created_at": now.isoformat(timespec="seconds"),
        "experiment_type": "automl",
        "backend": "autogluon",
        "model_name": selected_model,
        "task_type": task_type,
        "positive_class": str(positive_label) if positive_label is not None else None,
        "data_version": data_version,
        "config_hash": config_hash,
        "primary_metric": primary_metric,
        "auxiliary_metrics": [metric for metric in metric_names if metric != primary_metric],
        "metrics": {k: v for k, v in metrics.items() if k != "classification_report"},
        "test_metrics": {k: v for k, v in test_metrics.items() if k != "classification_report"},
        "artifact_path": str(artifact_dir),
        "experiment_path": str(run_dir),
        "confusion_matrix_path": str(confusion_matrix_path) if confusion_matrix_path else None,
        "threshold_metrics_path": str(threshold_metrics_path) if threshold_metrics_path else None,
        "leaderboard_path": str(leaderboard_path),
        "automl_summary_path": str(automl_summary_path),
        "limitations": "AutoGluon pipeline candidate. Validation/tuning data is not fully unseen; test metrics are final-check context.",
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
                "leaderboard_path": str(leaderboard_path),
                "automl_summary_path": str(automl_summary_path),
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
    parser.add_argument("--decisions", type=str, default=None, help="training_decisions YAML block이 있는 Markdown 파일")
    parser.add_argument("--data", type=str, default=None, help="학습용 processed CSV 경로")
    parser.add_argument("--target", type=str, default=None, help="target column 이름")
    parser.add_argument("--data-version", type=str, default=None, help="실험에 사용할 data_version")
    parser.add_argument("--task-type", choices=["classification", "regression"], default=None, help="task 종류")
    parser.add_argument("--positive-class", type=str, default=None, help="binary classification positive class")
    parser.add_argument("--primary-metric", type=str, default=None, help="AutoGluon이 최적화할 primary metric")
    parser.add_argument("--metrics", type=str, default=None, help="보조 metric 목록. 예: accuracy,macro_f1,roc_auc")
    parser.add_argument("--test-size", type=float, default=None, help="전체 데이터 중 test 비율")
    parser.add_argument("--val-size", type=float, default=None, help="전체 데이터 중 validation 비율")
    parser.add_argument("--no-stratify", action="store_true", help="stratified split을 끕니다")
    parser.add_argument("--time-limit", type=int, default=None, help="AutoGluon 탐색 시간 제한(초)")
    parser.add_argument("--presets", type=str, default=None, help="AutoGluon preset. 기본: medium_quality")
    args = parser.parse_args()
    try:
        main(args.config, args)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        parser.exit(1, f"error: {exc}\n")
