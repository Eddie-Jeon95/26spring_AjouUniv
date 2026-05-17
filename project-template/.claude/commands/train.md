---
description: "baseline과 일반 실험을 계획, 실행, 기록합니다."
argument-hint: "[experiment goal or baseline]"
disable-model-invocation: true
---

# /train

processed CSV와 `training_decisions`를 기준으로 baseline 또는 새 실험을 실행하고 결과를 기록합니다.

$ARGUMENTS

## 1. 확인할 기준

- `PROJECT_GUIDE.md`의 Modeling, Metrics, Experiment 기준
- `reports/PROJECT_REPORT.md`의 Training Decisions와 Experiment Report
- `data_manifest.json`의 data_version
- `model_registry.json`의 기존 baseline/실험

## 2. 학생 Decision Gate

실행 전에 아래 결정을 학생이 확인하게 합니다.

| 항목 | 기본 추천 | 선택지 |
|------|-----------|--------|
| baseline model | classification: logistic_regression, regression: linear_regression | random_forest 계열 포함 |
| primary metric | 문제 정의 기준 | classification/regression 지원 metric |
| auxiliary metrics | 1~3개 | 문제 비용과 연결 |
| split | Data Card 추천 | random/stratified/time/group 가능 여부 |
| 실험 가설 | baseline이면 단순 비교 기준 | 새 실험이면 변경점과 기대 효과 |

## 3. 실행

`reports/PROJECT_REPORT.md`의 `training_decisions` block을 채운 뒤 실행합니다.

```bash
python scripts/train.py --decisions reports/PROJECT_REPORT.md
```

임시 실험만 CLI override를 허용합니다. 예: `--model-name random_forest`.

## 4. 기록

실행 후 아래를 확인하고 Experiment Report 섹션에 요약합니다.

- run_id와 model_id
- validation metrics와 test metrics
- confusion matrix 또는 residual 확인 포인트
- threshold metrics 생성 여부
- baseline 대비 비교 가능 여부
- 실패해도 배운 점과 다음 실험 후보
