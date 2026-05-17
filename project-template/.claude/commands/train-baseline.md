---
description: "processed CSV로 task/metric 기준을 고정하고 baseline 학습을 계획, 실행, 기록합니다."
argument-hint: "[processed_path] target=[target] data_version=[version] [task_type=classification|regression] [primary_metric=...] [test_size=0.2] [val_size=0.2]"
disable-model-invocation: true
---

# /train-baseline

processed CSV로 baseline 학습을 실행하기 전에 계획을 짧게 정리하고, 실행 후 결과를 `reports/EXPERIMENT_REPORT.md`에 이어서 정리하는 커맨드입니다.

---

입력 인자 예시:

```text
/train-baseline data/processed/[processed_file] target=[target] data_version=[data_version]
/train-baseline data/processed/[processed_file] target=[target] data_version=[data_version] test_size=0.2 val_size=0.2
```

$ARGUMENTS

## 1단계: 학습 전 확인

- `CLAUDE.md`의 모델링 원칙 확인
- `docs/specs/MODELING_SPEC.md`의 baseline 기준 확인
- `docs/specs/METRICS_AND_INTERPRETATION_SPEC.md`의 metric 기준 확인
- `reports/DATA_CARD.md`에서 processed 파일, target, Feature Drop Candidates, 누수 의심, split 추천 확인
- `docs/specs/PROJECT_SPEC.md`의 task type, positive class, primary/auxiliary metric 확인
- processed CSV 파일이 존재하는지 확인
- target column이 파일에 있는지 확인
- Data Card에 기록된 drop 후보가 processed CSV에서 의도대로 제외됐는지 확인
- classification인지 regression인지 확인
- binary classification이면 positive class와 threshold 조정 가능 여부 확인
- `--primary-metric`이 실제 계산 가능한 metric인지 확인
- `DATA_CARD.md`의 split 추천과 `training_decisions.split`이 다르면, 실행 전에 학생에게 확인하고 결정값을 `EXPERIMENT_REPORT.md`에 반영합니다.
- 현재 스크립트로 맞출 수 없는 split이면 실행 전에 사용자에게 짧게 확인하고, 그대로 진행한다면 `EXPERIMENT_REPORT.md`에 불일치 이유를 기록합니다.

## 2단계: Baseline Plan 정리

학습 전에 `reports/EXPERIMENT_REPORT.md`의 `training_decisions` YAML block을 우선 확인하고,
학생이 선택한 processed 데이터, task type, metric, split, baseline 모델을 block에 반영하세요.

`Baseline Plan` 섹션에 옮길 수 있게 다음도 정리하세요.

```markdown
## Baseline Plan
- 사용할 processed 데이터:
- data_version:
- target column:
- task type / positive class:
- 사용할 feature 정책:
- 사용할 feature:
- 제외할 feature와 이유:
- 상관/중복 점검 결과:
- split 방식 / seed:
- primary metric / 보조 metric:
- baseline 모델:
- baseline을 선택한 이유:
- train 전에 확인한 누수 위험:
- 실행 명령:
```

## 3단계: 학습 실행

기본 실행은 MD decision block을 사용합니다.

```bash
python scripts/train.py --decisions reports/EXPERIMENT_REPORT.md
```

임시 override가 필요할 때만 CLI 인자를 함께 넘깁니다.
예: `--model-name random_forest`, `--primary-metric macro_f1`, `--test-size 0.2`.

## 4단계: 결과 기록

실행 후 다음을 확인합니다.

- 생성된 `run_id`
- `experiments/runs/<run_id>/config.yaml`
- `metrics.json` (validation metric, 실험 비교 기준)
- `test_metrics.json` (test metric, 최종 확인용)
- `predictions.csv` (validation split 기준)
- classification이면 `confusion_matrix.json` (validation split 기준)
- binary classification이면 `threshold_metrics.csv` 생성 여부
- `model_registry.json`의 새 record
- registry record의 `task_type`, `positive_class`, `auxiliary_metrics`, `threshold_metrics_path`
- run의 `config.yaml`에 저장된 split이 계획과 같은지 확인

`reports/EXPERIMENT_REPORT.md`에 옮길 수 있게 정리하세요.

```markdown
## Baseline Result
- baseline model_id:
- data_version:
- 주요 metric:
- 보조 metric:
- threshold 필요 여부:
- 비교 기준으로 삼을 이유:
- 한계:
- 다음 실험 후보:
```

---

주의:
- data_version이 다른 실험과 직접 우열을 말하지 않습니다.
- split이나 seed가 다른 실험도 직접 우열을 말하지 않습니다.
- accuracy만 보고 결론을 내리지 말고 class별 성능과 confusion matrix 확인을 제안하세요.
- regression에서는 `mae`, `rmse`, `r2`를 함께 보고 큰 오차 사례를 확인하세요.
- binary classification에서 threshold 조정은 validation 기준으로 검토하고 test set으로 선택하지 마세요.
- 실행에 사용된 effective config는 run 폴더의 `config.yaml`에 자동 저장됩니다.
- `model_registry.json`의 `metrics`는 validation 기준이고, `test_metrics`는 최종 확인용입니다.
- `predictions.csv`에 feature와 `original_index`가 있으면 오류 분석에 사용하고, 없으면 feature 값을 추측하지 말고 재현 가능한 split으로 다시 생성하거나 한계를 명시하세요.
