---
description: "AutoGluon 실행 전에 metric, split, leakage 제외 컬럼, 성공 기준을 고정합니다."
argument-hint: "[processed_path] target=[target] data_version=[version] [task_type=classification|regression] [primary_metric=...]"
disable-model-invocation: true
---

# /plan-automl

AutoGluon Tabular를 실행하기 전에 계획을 확정하는 커맨드입니다.
AutoML은 계획을 대신 세우지 않습니다. 이 단계에서 사람이 정한 조건 안에서만 전처리, 모델, 하이퍼파라미터 조합을 탐색합니다.

---

입력 인자 예시:

```text
/plan-automl data/processed/[processed_file] target=[target] data_version=[data_version] task_type=classification primary_metric=macro_f1
/plan-automl data/processed/[processed_file] target=[target] data_version=[data_version] task_type=regression primary_metric=rmse
```

$ARGUMENTS

## 1단계: 기준 문서 확인

- `docs/specs/PROJECT_SPEC.md`의 문제 정의, 오류 비용, metric decision 확인
- `docs/specs/DATA_ANALYSIS_SPEC.md`의 leakage 후보, split 추천, processed 데이터 기준 확인
- `docs/specs/MODELING_SPEC.md`의 AutoGluon AutoML 원칙 확인
- `docs/specs/METRICS_AND_INTERPRETATION_SPEC.md`의 metric 해석 기준 확인
- `reports/DATA_CARD.md`의 target, Feature Drop Candidates, 제외 컬럼, split/seed, processed 파일 확인

## 2단계: Baseline 결과 확인

- `model_registry.json`에서 같은 `data_version`의 baseline record가 있는지 확인
- baseline의 `primary_metric`, `auxiliary_metrics`, `task_type`, `positive_class` 확인
- `experiments/runs/<run_id>/metrics.json`과 `test_metrics.json` 확인
- binary classification이면 `threshold_metrics.csv` 존재 여부와 threshold 조정 필요성을 확인
- baseline이 아직 없으면 AutoML 실행 전에 `/train-baseline`을 먼저 제안

## 3단계: AutoML Plan 작성

`reports/EXPERIMENT_REPORT.md`의 `AutoML Plan` 섹션에 옮길 수 있게 아래를 정리합니다.

```markdown
## AutoML Plan
- backend: AutoGluon Tabular
- 실행 목적:
- processed 데이터:
- data_version:
- target column:
- task type:
- positive class:
- leakage 제외 컬럼:
- feature drop 근거:
- split 방식 / seed:
- primary metric:
- auxiliary metrics:
- baseline 기준:
- 성공 기준:
- test set 사용 원칙: 최종 확인용, 모델 선택/threshold 선택에는 미사용
- time_limit / presets:
- 예상 한계:
- 실행 명령:
```

## 4단계: 실행 명령 제안

YAML 직접 수정보다 CLI 인자를 우선합니다.

```bash
python scripts/train_automl.py \
  --data data/processed/[processed_file] \
  --target [target] \
  --data-version [data_version] \
  --task-type classification \
  --primary-metric macro_f1 \
  --metrics accuracy,precision_macro,recall_macro \
  --test-size 0.2 \
  --val-size 0.2 \
  --time-limit 300 \
  --presets medium_quality
```

regression이면 예를 들어 아래처럼 실행합니다.

```bash
python scripts/train_automl.py \
  --data data/processed/[processed_file] \
  --target [target] \
  --data-version [data_version] \
  --task-type regression \
  --primary-metric rmse \
  --metrics mae,rmse,r2
```

## 5단계: 실행 후 확인 항목

- `experiments/runs/<run_id>/leaderboard.csv`
- `experiments/runs/<run_id>/automl_summary.json`
- `experiments/runs/<run_id>/metrics.json`
- `experiments/runs/<run_id>/test_metrics.json`
- binary classification이면 `experiments/runs/<run_id>/threshold_metrics.csv`
- `models/automl/<run_id>/` AutoGluon artifact
- `model_registry.json`의 `experiment_type=automl`, `backend=autogluon`, `leaderboard_path`, `automl_summary_path`

---

주의:
- AutoGluon에는 train과 validation만 전달하고, test set은 최종 확인용으로만 사용합니다.
- 같은 `data_version`, split, seed, metric이 아니면 baseline과 직접 우열을 말하지 않습니다.
- global feature importance, global SHAP, batch prediction은 현재 범위에서 제외합니다.
- 단일 입력 row에 대한 AutoGluon 내부 모델 2개 local SHAP explanation은 Prediction 탭에서 확인할 수 있습니다.
- AutoGluon 결과가 좋아도 leakage 후보 컬럼과 validation/test 차이를 먼저 확인합니다.
