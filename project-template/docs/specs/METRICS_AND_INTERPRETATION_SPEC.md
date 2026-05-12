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

---

## 5. Error Analysis 기준

오류 분석은 다음 순서로 합니다.

1. confusion matrix 또는 residual 분포를 확인합니다.
2. 오답 샘플을 직접 봅니다.
3. 실패 유형을 2~5개로 묶습니다.
4. 각 유형마다 가능한 원인과 다음 실험을 연결합니다.
5. 개선된 모델이 어떤 실패 유형을 줄였는지 확인합니다.

classification 실험에서는 `experiments/runs/<run_id>/confusion_matrix.json`을 Streamlit에서 다시 볼 수 있게 남깁니다.
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

## 7. 발표 전 점검

- [ ] 주요 metric 선택 이유를 설명할 수 있다.
- [ ] baseline과 최종 모델을 같은 data_version에서 비교했다.
- [ ] 좋아진 metric과 나빠진 metric을 모두 언급했다.
- [ ] error analysis 결과가 다음 실험과 연결된다.
- [ ] 모델 한계가 `MODEL_CARD.md`에 기록됐다.
