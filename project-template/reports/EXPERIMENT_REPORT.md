# Experiment Report

## 1. Baseline Plan

- **사용할 processed 데이터**:
- **data_version**:
- **target column**:
- **task type / positive class**:
- **사용할 feature**:
- **제외할 feature와 이유**:
- **split 방식 / seed**:
- **primary metric / 보조 metric**:
- **baseline 모델**:
- **baseline을 선택한 이유**:
- **train 전에 확인한 누수 위험**:
- **실행 명령**:

```bash
python scripts/train.py --data data/processed/[file].csv --target [target] --data-version [version]
```

## 2. Baseline Result

- **baseline model_id**:
- **data_version**:
- **주요 metric**:
- **보조 metric trade-off**:
- **threshold 필요 여부**:
- **비교 기준으로 삼을 이유**:
- **한계**:

## 3. AutoML Plan

- **AutoML backend**: AutoGluon Tabular
- **실행 목적**: baseline 이후 같은 데이터/split/metric 조건에서 pipeline 후보 비교
- **processed 데이터 / data_version**:
- **target / task type / positive class**:
- **고정할 split / seed**:
- **primary metric / auxiliary metrics**:
- **leakage 제외 컬럼**:
- **test set 사용 원칙**: 모델 선택과 threshold 선택에는 사용하지 않고 최종 확인에만 사용
- **성공 기준**:
- **예상 한계 / 비용**:
- **실행 명령**:

```bash
python scripts/train_automl.py --data data/processed/[file].csv --target [target] --data-version [version]
```

## 4. AutoML Result

- **automl model_id**:
- **selected AutoGluon model**:
- **leaderboard_path**:
- **automl_summary_path**:
- **validation 주요 metric**:
- **test 주요 metric**:
- **baseline 대비 개선/악화**:
- **한계와 채택 보류 이유**:

## 5. 실험 요약

| run_id | type | 가설/목적 | 변경점 | 주요 metric | 결과 해석 | 다음 액션 |
|--------|------|-----------|--------|-------------|-----------|-----------|
|        |      |           |        |             |           |           |

모델별 metric 추이와 confusion matrix는 Streamlit 대시보드에서 다시 확인할 수 있도록 `model_registry.json`과 `experiments/runs/<run_id>/confusion_matrix.json`을 함께 사용합니다.
AutoGluon 실험은 `leaderboard.csv`와 `automl_summary.json`도 함께 확인합니다.

## 6. 모델 비교

- 가장 설득력 있는 모델:
- 선택 이유:
- trade-off:
- 아직 불확실한 점:

## 7. 개선 근거

- 어떤 오류 유형을 줄였는가?
- 어떤 feature 또는 모델 변경이 효과적이었는가?
- metric 외에 확인한 근거는 무엇인가?

## 8. 실패한 실험에서 배운 점

-

## 9. 다음 실험 후보

- [ ]
- [ ]
- [ ]
