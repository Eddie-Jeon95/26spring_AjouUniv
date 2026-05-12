# /train-baseline

processed CSV로 baseline 학습을 실행하기 전에 계획을 짧게 정리하고, 실행 후 결과를 `reports/EXPERIMENT_REPORT.md`에 이어서 정리하는 커맨드입니다.

---

입력 인자 예시:

```text
/train-baseline data/processed/banknote_v1.csv target=label data_version=banknote-v1
/train-baseline data/processed/banknote_v1.csv target=label data_version=banknote-v1 test_size=0.2 val_size=0.2
```

$ARGUMENTS

## 1단계: 학습 전 확인

- `CLAUDE.md`의 모델링 원칙 확인
- `docs/specs/MODELING_SPEC.md`의 baseline 기준 확인
- `docs/specs/METRICS_AND_INTERPRETATION_SPEC.md`의 metric 기준 확인
- `reports/DATA_CARD.md`에서 processed 파일, target, 누수 의심, split 추천 확인
- processed CSV 파일이 존재하는지 확인
- target column이 파일에 있는지 확인
- `DATA_CARD.md`의 split 추천과 `configs/default.yaml`의 split 기본값이 다르면, 가능한 경우 CLI 인자(`--test-size`, `--val-size`, `--no-stratify`)로 Data Card 추천을 맞춥니다.
- CLI로 맞출 수 없는 split이면 실행 전에 사용자에게 짧게 확인하고, 그대로 진행한다면 `EXPERIMENT_REPORT.md`에 불일치 이유를 기록합니다.

## 2단계: Baseline Plan 정리

학습 전에 `reports/EXPERIMENT_REPORT.md`의 `Baseline Plan` 섹션에 옮길 수 있게 다음을 정리하세요.

```markdown
## Baseline Plan
- 사용할 processed 데이터:
- data_version:
- target column:
- 사용할 feature:
- 제외할 feature와 이유:
- split 방식 / seed:
- primary metric / 보조 metric:
- baseline 모델:
- baseline을 선택한 이유:
- train 전에 확인한 누수 위험:
- 실행 명령:
```

## 3단계: 학습 실행

YAML 직접 수정을 요구하지 말고 CLI 인자로 실행합니다.

```bash
python scripts/train.py \
  --data data/processed/banknote_v1.csv \
  --target label \
  --data-version banknote-v1 \
  --test-size 0.2 \
  --val-size 0.2
```

입력 인자에 `test_size=...`, `val_size=...`, `stratify=false`가 있으면 각각 `--test-size`, `--val-size`, `--no-stratify`로 변환해 실행합니다.

## 4단계: 결과 기록

실행 후 다음을 확인합니다.

- 생성된 `run_id`
- `experiments/runs/<run_id>/config.yaml`
- `metrics.json`
- `predictions.csv`
- `confusion_matrix.json`
- `model_registry.json`의 새 record
- run의 `config.yaml`에 저장된 split이 계획과 같은지 확인

`reports/EXPERIMENT_REPORT.md`에 옮길 수 있게 정리하세요.

```markdown
## Baseline Result
- baseline model_id:
- data_version:
- 주요 metric:
- 비교 기준으로 삼을 이유:
- 한계:
- 다음 실험 후보:
```

---

주의:
- data_version이 다른 실험과 직접 우열을 말하지 않습니다.
- split이나 seed가 다른 실험도 직접 우열을 말하지 않습니다.
- accuracy만 보고 결론을 내리지 말고 class별 성능과 confusion matrix 확인을 제안하세요.
- 실행에 사용된 effective config는 run 폴더의 `config.yaml`에 자동 저장됩니다.
- `predictions.csv`에 feature와 `original_index`가 있으면 오류 분석에 사용하고, 없으면 feature 값을 추측하지 말고 재현 가능한 split으로 다시 생성하거나 한계를 명시하세요.
