---
description: "여러 실험의 metric, split, data_version, trade-off를 비교합니다."
argument-hint: "[data_version 선택사항]"
disable-model-invocation: true
---

# /compare-models

여러 실험의 성능과 trade-off를 비교하는 커맨드입니다.

---

`model_registry.json`과 `experiments/runs/`의 결과를 바탕으로 모델을 비교해주세요.

## 1단계: 실험 목록 정리

- `CLAUDE.md`에서 모델 비교 원칙 확인
- `docs/specs/METRICS_AND_INTERPRETATION_SPEC.md`에서 metric 해석 기준 확인
- `model_registry.json`을 읽고 model_id, data_version, metric, artifact path를 확인
- 각 record의 `experiment_type`, `backend`, `task_type`, `positive_class`, `primary_metric`, `auxiliary_metrics` 확인
- 각 run의 `confusion_matrix.json`이 있으면 주요 혼동 패턴도 확인
- binary classification이면 `threshold_metrics.csv` 존재 여부 확인
- AutoGluon record이면 `leaderboard_path`, `automl_summary_path`를 읽고 selected model, eval_metric, time_limit, presets 확인
- 각 run의 `config.yaml`에서 split, seed, target, feature 구성이 같은지 확인
- 같은 data_version끼리 우선 비교
- split이나 data_version이 다른 실험은 직접 비교가 어렵다고 표시
- `mae`, `rmse`는 낮을수록 좋고, 나머지 기본 metric은 높을수록 좋다고 해석

## 2단계: 비교 표 작성

| model_id | type/backend | data_version | split/seed | primary metric | 보조 metric | 변경점 | 장점 | 한계 |
|----------|--------------|--------------|------------|----------------|-----------|--------|------|------|
|          |              |              |            |                |           |        |      |      |

AutoGluon 결과가 있으면 별도 표로 요약합니다.

| rank | model | score | fit_time | pred_time | notes |
|------|-------|-------|----------|-----------|-------|
|      |       |       |          |           |       |

## 3단계: 해석

```markdown
## 결론
- 현재 가장 설득력 있는 모델:
- 이유:
- baseline 대비 개선/악화:
- AutoGluon 채택 여부:
- 아직 불확실한 점:
- spec 기준에서 부족한 근거:

## 다음 실험 제안
1.
2.
3.
```

## 4단계: 모델 비교 요약

metric 숫자 나열이 아니라, 왜 모델이 개선됐는지 설명하는 3~5문장 요약을 제안해주세요.

---

주의:
- data_version이 다르면 그 차이를 반드시 언급하세요.
- 같은 data_version이라도 split, seed, feature 구성이 다르면 직접 우열 비교를 보류하세요.
- accuracy만 좋아지고 특정 class recall이 나빠진 경우 경고하세요.
- AutoGluon은 전처리 + 모델 + 하이퍼파라미터 조합을 비교한 결과이므로 모델 이름만 보고 결론내리지 마세요.
- test metric은 최종 확인용이며 모델 선택의 주 근거는 validation metric과 문제 정의입니다.
- global feature importance, global SHAP, batch prediction은 현재 비교 범위에서 제외합니다.
- Prediction 탭의 local SHAP explanation은 단일 row에 대한 모델 판단 근거 후보이며, 인과 설명으로 해석하지 마세요.
