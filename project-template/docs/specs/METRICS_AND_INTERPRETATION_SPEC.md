# Metrics and Interpretation Spec

metric은 모델을 고르는 숫자일 뿐 아니라, 문제를 어떻게 바라보는지 보여주는 언어입니다.
항상 문제 정의와 오류 비용에 맞춰 해석하세요.

---

## 1. Metric 선택 기준

먼저 오류 비용을 정합니다.

- false positive가 더 위험한가?
- false negative가 더 위험한가?
- class별 성능 균형이 중요한가?
- 큰 오차가 작은 오차보다 훨씬 위험한가?
- 예측 확률의 calibration이 중요한가?

그 다음 주요 metric 1개와 보조 metric 1~3개를 정합니다.
`configs/default.yaml`의 `tracking.primary_metric`과 `tracking.auxiliary_metrics`에 같은 이름으로 남기고, CLI 실행 시에는 `--primary-metric`, `--metrics`로 덮어쓸 수 있습니다.

템플릿 내부 metric 이름:

| Task | 지원 metric |
|------|-------------|
| classification | `accuracy`, `macro_f1`, `precision_macro`, `recall_macro`, `roc_auc`, `pr_auc` |
| regression | `mae`, `rmse`, `r2` |

`--primary-metric`은 실제 계산되어 `metrics.json`에 저장되는 metric이어야 합니다.
예를 들어 binary probability가 없는 모델에서 `roc_auc`를 primary metric으로 쓰면 실행 단계에서 명확한 에러가 나야 합니다.

---

## 2. Classification 기본 지표

| 상황 | 주요 metric 후보 | 함께 볼 것 |
|------|------------------|------------|
| class 균형 | accuracy | confusion matrix |
| class imbalance | macro F1, balanced accuracy | class별 precision/recall |
| positive 탐지가 중요 | recall | false positive 수 |
| 오탐 비용이 큼 | precision | false negative 수 |
| threshold 조정 가능 | ROC-AUC, PR-AUC | threshold별 precision/recall |

accuracy는 class imbalance가 있으면 쉽게 착시를 만듭니다.
binary classification에서 `positive_class`와 probability가 있으면 `threshold_metrics.csv`를 만들어 threshold별 precision, recall, F1, TP/FP/TN/FN을 확인합니다.
threshold는 validation split에서 검토하고, test set으로 선택하지 않습니다.

---

## 3. Regression 기본 지표

| 상황 | 주요 metric 후보 | 해석 |
|------|------------------|------|
| 평균 오차 크기 | MAE | 실제 단위로 설명 쉬움 |
| 큰 오차를 강하게 벌점 | RMSE | outlier에 민감 |
| 상대적 설명력 | R2 | 단독 사용 주의 |
| 비율 오차 중요 | MAPE | 0 근처 값 주의 |

regression은 residual plot과 큰 오차 사례를 함께 확인합니다.

---

## 4. 모델 비교 기준

모델 비교 시 반드시 확인합니다.

- 같은 data_version인가?
- 같은 split과 seed인가?
- 같은 metric으로 계산됐는가?
- 주요 metric과 보조 metric이 같은 방향으로 좋아졌는가?
- 성능 차이가 해석 가능한 변경점과 연결되는가?
- 더 복잡한 모델을 쓸 만큼 이득이 충분한가?

data_version이나 split이 다르면 표에는 넣어도 직접 우열을 말하지 않습니다.
`mae`, `rmse`는 낮을수록 좋고, 나머지 기본 metric은 높을수록 좋습니다.

AutoGluon 비교 시 추가 확인:

- AutoGluon은 전처리 + 모델 + 하이퍼파라미터 조합을 탐색한 결과이므로 baseline과 단순히 “모델 이름”만 비교하지 않습니다.
- AutoGluon의 primary metric 최적화가 auxiliary metric을 악화시키지 않았는지 확인합니다.
- validation 성능만으로 결론내지 말고 test metric은 최종 확인용으로 따로 해석합니다.
- test set이 AutoGluon 탐색이나 threshold 선택에 사용되지 않았는지 확인합니다.
- AutoGluon metric 이름은 내부 이름과 다를 수 있습니다. 내부 `macro_f1`은 AutoGluon의 `f1_macro`, `pr_auc`는 `average_precision`, `rmse`는 `root_mean_squared_error`로 매핑합니다.
- Prediction 탭의 local SHAP explanation은 같은 입력 row에 대해 AutoGluon 내부 모델 2개의 판단 근거 후보를 비교하는 용도입니다. feature 기여도처럼 읽되 인과 설명으로 쓰지 않습니다.
- 대안으로 LIME은 한 샘플 주변의 단순 surrogate 모델을 맞추는 방식이라 직관적이지만 perturbation 설정에 따라 불안정할 수 있습니다. DiCE/counterfactual은 “왜 이런 예측인가”보다 “무엇을 바꾸면 예측이 바뀌는가”에 가깝기 때문에 현재 MVP에서는 제외합니다.

---

## 5. Error Analysis 기준

오류 분석은 다음 순서로 합니다.

1. confusion matrix 또는 residual 분포를 확인합니다.
2. 오답 샘플을 직접 봅니다.
3. 실패 유형을 2~5개로 묶습니다.
4. 각 유형마다 가능한 원인과 다음 실험을 연결합니다.
5. 개선된 모델이 어떤 실패 유형을 줄였는지 확인합니다.

classification 실험에서는 `experiments/runs/<run_id>/confusion_matrix.json`을 Streamlit에서 다시 볼 수 있게 남깁니다.
binary classification에서 `threshold_metrics.csv`가 있으면 낮은 recall/precision 문제가 threshold 조정으로 완화되는지 확인합니다.
regression 실험에서는 `mae`, `rmse`, `r2`와 함께 큰 오차 샘플을 확인합니다.
오답 feature를 해석할 때는 `predictions.csv`에 저장된 feature 또는 재현 가능한 test split만 근거로 사용합니다.
반올림된 출력이나 row 번호만 보고 원본 feature, 중복, 누수를 단정하지 않습니다.
중복 누수를 의심할 때는 processed 파일에서 exact duplicate와 feature-only duplicate 수를 직접 계산한 뒤 말합니다.

기록 예시:

| 실패 유형 | 근거 | 원인 가설 | 다음 실험 |
|-----------|------|-----------|-----------|
| minority class 미탐 | recall 낮음 | class imbalance | class weight 적용 |
| 긴 입력에서 오류 | 길이별 성능 하락 | truncation | max length 조정 |

---

## 6. 결과 해석 문장 기준

좋은 해석:

```markdown
Baseline 대비 macro F1이 0.62에서 0.70으로 올랐고, 특히 minority class recall이 개선됐다.
EDA에서 확인한 class imbalance를 class weight로 보정한 결과라 문제 정의와 연결된다.
다만 false positive가 늘어 실제 사용 시 threshold 조정이 필요하다.
```

피해야 할 해석:

```markdown
성능이 올랐으므로 좋은 모델이다.
```

---

## 7. 결과 해석 전 점검

- [ ] 주요 metric 선택 이유를 설명할 수 있다.
- [ ] baseline과 최종 모델을 같은 data_version에서 비교했다.
- [ ] 좋아진 metric과 나빠진 metric을 모두 언급했다.
- [ ] AutoGluon 결과라면 같은 split/metric 조건에서 baseline과 비교했다.
- [ ] binary threshold를 조정했다면 validation 기준으로 선택했고 test set을 쓰지 않았다.
- [ ] error analysis 결과가 다음 실험과 연결된다.
- [ ] 모델 한계가 `MODEL_CARD.md`에 기록됐다.
