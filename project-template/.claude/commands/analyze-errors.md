---
description: "최근 실험의 오류 유형을 분석하고 다음 개선 실험을 제안합니다."
argument-hint: "[run_id 또는 model_id]"
disable-model-invocation: true
---

# /analyze-errors

모델의 실패 케이스를 분석하고 다음 개선 실험을 찾는 커맨드입니다.

---

최근 실험의 예측 결과와 metric을 보고 error analysis를 도와주세요.

## 1단계: 입력 파일 확인

- `docs/specs/DATA_ANALYSIS_SPEC.md`에서 데이터 품질/누수 기준 확인
- `docs/specs/METRICS_AND_INTERPRETATION_SPEC.md`에서 error analysis 기준 확인
- `model_registry.json`에서 분석할 run_id 확인
- registry record의 `task_type`, `positive_class`, `primary_metric`, `threshold_metrics_path`, `experiment_type`, `backend` 확인
- 해당 `experiments/runs/<run_id>/metrics.json` 읽기
- 가능하면 `experiments/runs/<run_id>/confusion_matrix.json` 읽기
- binary classification이면 가능하면 `experiments/runs/<run_id>/threshold_metrics.csv` 읽기
- AutoGluon record이면 가능하면 `leaderboard.csv`, `automl_summary.json` 읽기
- 가능하면 `experiments/runs/<run_id>/predictions.csv` 읽기
- 가능하면 `experiments/runs/<run_id>/config.yaml`을 읽어 data path, split, seed를 확인
- `predictions.csv`에 `original_index`와 feature columns가 있는지 확인
- 데이터 원본이나 샘플이 없으면, 현재 가능한 범위와 한계를 먼저 말하기
- `predictions.csv`에 feature가 없으면 feature 값을 추측하지 말고, `config.yaml`의 data path와 split seed로 test split을 재현하거나 “현재 파일만으로는 feature 기반 오류 원인 분석 불가”라고 명시
- 중복/누수 의심을 말하기 전에는 processed 파일에서 `df.duplicated().sum()`과 feature-only duplicate 수를 실제로 계산

## 2단계: 실패 유형화

classification이면 confusion matrix, class별 precision/recall/F1, threshold별 precision/recall trade-off를 확인합니다.
regression이면 `mae`, `rmse`, `r2`, 큰 residual 사례를 중심으로 확인합니다.

```markdown
## Error Types

| 유형 | 관찰 근거 | 가능한 원인 | 확인할 추가 분석 |
|------|-----------|-------------|------------------|
|      |           |             |                  |
```

## 3단계: 개선 후보 제안

- 데이터 품질 개선
- feature engineering
- metric 또는 threshold 조정
- baseline 모델/하이퍼파라미터 변경
- baseline 분석 뒤 필요하면 `/plan-automl`로 AutoGluon 비교 실험 계획
- 추가 수집이나 라벨 재검토

## 4단계: 리포트 반영

`reports/ERROR_ANALYSIS.md`에 옮길 수 있는 요약 문장을 제안해주세요.

## 5단계: Spec 기준 점검

- metric 해석 기준을 만족한 점:
- 아직 부족한 분석:
- 다음 실험과 연결되는 개선 후보:

---

주의:
- 개인정보나 민감한 원본 샘플은 그대로 출력하지 마세요.
- 근거 없이 모델 변경만 제안하지 말고, 어떤 실패 유형을 줄이려는지 연결하세요.
- binary classification의 threshold 후보는 validation 기준으로만 제안하고 test set으로 선택하지 마세요.
- AutoGluon 결과 분석 시 pipeline 복잡도와 leakage 후보 feature 위험을 함께 언급하세요.
- 화면에 반올림되어 같은 값처럼 보이는 행을 exact duplicate로 단정하지 마세요.
- processed 데이터의 exact duplicate가 0개라면 “잔여 중복 가능성”을 다음 실험 근거로 제안하지 마세요.
