---
description: "모델 비교, error analysis, 최종 모델 선택 근거를 정리합니다."
argument-hint: "[run_id or comparison goal]"
disable-model-invocation: true
---

# /evaluate

여러 run을 비교하고, 실패 유형과 최종 모델 선택 근거를 `PROJECT_REPORT.md`에 정리합니다.

$ARGUMENTS

## 1. 확인할 기준

- `PROJECT_GUIDE.md`의 Model Comparison, Error Analysis 기준
- `model_registry.json`
- `experiments/runs/<run_id>/metrics.json`
- `test_metrics.json`, `predictions.csv`, `confusion_matrix.json`, `leaderboard.csv`

## 2. 비교 원칙

- 같은 data_version, split, seed, metric만 직접 우열을 말합니다.
- validation metric은 실험 비교 기준입니다.
- test metric은 최종 확인용입니다.
- metric 하나만 보지 말고 auxiliary metric, error type, 복잡도, leakage 위험을 함께 봅니다.

## 3. 학생 Decision Gate

아래 표로 선택 근거를 확인합니다.

| 후보 모델 | 좋아진 점 | 나빠진 점 | 리스크 | 학생 결정 |
|-----------|-----------|-----------|--------|-----------|
| baseline | | | | |
| automl/experiment | | | | |

## 4. 기록

`reports/PROJECT_REPORT.md`의 Error Analysis, Model Card, Final Checklist를 갱신합니다.

- 주요 실패 유형 2~5개
- 원인 가설과 다음 실험
- 최종 모델 선택 이유
- 운영 한계와 사용 금지 조건
- test set을 모델 선택에 쓰지 않았다는 점검
