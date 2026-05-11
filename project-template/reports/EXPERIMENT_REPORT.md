# Experiment Report

## 1. Baseline Plan

- **사용할 processed 데이터**:
- **data_version**:
- **target column**:
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
- **비교 기준으로 삼을 이유**:
- **한계**:

## 3. 실험 요약

| run_id | 가설 | 변경점 | 주요 metric | 결과 해석 | 다음 액션 |
|--------|------|--------|-------------|-----------|-----------|
|        |      |        |             |           |           |

모델별 metric 추이와 confusion matrix는 Streamlit 대시보드에서 다시 확인할 수 있도록 `model_registry.json`과 `experiments/runs/<run_id>/confusion_matrix.json`을 함께 사용합니다.

## 4. 모델 비교

- 가장 설득력 있는 모델:
- 선택 이유:
- trade-off:
- 아직 불확실한 점:

## 5. 개선 근거

- 어떤 오류 유형을 줄였는가?
- 어떤 feature 또는 모델 변경이 효과적이었는가?
- metric 외에 확인한 근거는 무엇인가?

## 6. 실패한 실험에서 배운 점

-

## 7. 다음 실험 후보

- [ ]
- [ ]
- [ ]
