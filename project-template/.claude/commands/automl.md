---
description: "AutoGluon AutoML 실행 조건을 고정하고 결과를 기록합니다."
argument-hint: "[processed_path or data_version]"
disable-model-invocation: true
---

# /automl

AutoGluon은 사람이 정한 metric/split/leakage 제약 안에서 pipeline 후보를 비교하는 단계입니다.

$ARGUMENTS

## 1. 확인할 기준

- `PROJECT_GUIDE.md`의 AutoML, Metrics, Leakage 기준
- `reports/PROJECT_REPORT.md`의 `training_decisions`
- 같은 data_version의 baseline record
- processed CSV에서 leakage/drop 후보가 제외됐는지

baseline이 없으면 먼저 `/train baseline`을 제안합니다.

## 2. 학생 Decision Gate

AutoML 실행 전에 아래를 확정합니다.

| 항목 | 확인 내용 |
|------|-----------|
| 실행 목적 | baseline 개선 / pipeline 후보 비교 / 한계 확인 |
| primary metric | baseline과 같은 기준인지 |
| auxiliary metrics | 악화되면 안 되는 지표 |
| time_limit | 수업/실습 시간 안에 가능한지 |
| presets | 속도와 품질 trade-off |
| 성공 기준 | 얼마만큼 좋아져야 채택할지 |
| test set 원칙 | 최종 확인용, 모델 선택에는 미사용 |

## 3. 실행

`reports/PROJECT_REPORT.md`의 `training_decisions.automl`을 확인한 뒤 실행합니다.

```bash
python scripts/train_automl.py --decisions reports/PROJECT_REPORT.md
```

## 4. 기록

AutoML Result 섹션에 아래를 남깁니다.

- selected AutoGluon model
- leaderboard path
- automl_summary path
- validation/test metric 차이
- baseline 대비 좋아진 점과 나빠진 점
- pipeline 복잡도와 데모 가능성
