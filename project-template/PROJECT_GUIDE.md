# Project Guide

이 문서는 프로젝트 판단 기준을 한곳에 모은 가이드입니다.
세부 진행은 slash command를 사용하고, 실제 기록은 `reports/PROJECT_REPORT.md`에 남깁니다.

---

## 1. Project Definition

프로젝트 시작 시 아래 결정을 먼저 고정합니다.

- 입력 데이터와 예측 대상
- target column
- task type: classification / regression
- binary classification의 positive class
- 실제 사용자가 더 피해야 하는 오류: false positive / false negative / 큰 오차
- primary metric 1개와 auxiliary metrics 1~3개
- Streamlit 최종 화면에서 확인할 항목

목표 수치가 없으면 baseline 실행 후 "baseline 대비 개선"으로 성공 기준을 잡습니다.

---

## 2. Data Analysis

EDA는 모델링 전에 데이터가 믿을 만한지 판단하기 위한 단계입니다.

최소 확인 항목:

- row/column 수와 feature 타입
- target 분포
- 결측치, 중복, 이상치
- numeric/category/text/date feature 분포
- constant, near-constant, high-missing, ID-like feature
- numeric high-correlation pair
- target 이후 생성되거나 예측 시점에 알 수 없는 leakage 후보
- split 방식과 seed

drop 후보는 무조건 제거하지 않습니다.
학생에게 `keep`, `drop`, `investigate` 중 하나를 선택하게 하고, 근거를 `PROJECT_REPORT.md`에 남깁니다.

---

## 3. Processed Data

`data/processed/`에는 모델 학습용으로 정돈된 CSV를 저장합니다.
전처리 결정은 `reports/PROJECT_REPORT.md`의 `pipeline_decisions` block에 기록합니다.

processed 파일에 반영해도 되는 작업:

- 컬럼명 strip, rename
- 사용하지 않을 컬럼 drop
- EDA에서 근거를 남긴 leakage, ID-like, constant, high-missing, high-correlation 후보 제거
- target label mapping
- 중복 row 제거
- target 결측 row 제거
- 명확한 타입과 단위 정리

processed 파일에 미리 반영하지 말아야 하는 작업:

- scaler fit
- encoder fit
- imputer fit
- train/test 전체를 보고 fit하거나 확정한 supervised feature selection
- test set 정보를 이용한 threshold/model 선택

fit이 필요한 변환은 train split 이후 pipeline 안에서 처리합니다.

---

## 4. Modeling

baseline은 단순하고 빠르게 실행되어야 하며 이후 실험의 비교 기준입니다.

권장 baseline:

| 문제 유형 | baseline |
|-----------|----------|
| classification | Logistic Regression, RandomForest |
| regression | Linear Regression, RandomForestRegressor |

새 실험은 아래 문장을 채울 수 있어야 합니다.

```markdown
[변경점]을 적용하면 [현재 관찰한 문제]가 줄어들 것이고,
그 결과 [metric 또는 error type]이 개선될 것이다.
```

실험 조건은 `reports/PROJECT_REPORT.md`의 `training_decisions` block에 기록합니다.
config YAML 파일을 직접 수정하지 않고 `--decisions reports/PROJECT_REPORT.md`를 기본 실행 방식으로 사용합니다.

---

## 5. Metrics

metric은 모델을 고르는 숫자이면서 문제를 어떻게 바라보는지 보여주는 언어입니다.

지원 metric:

| Task | Metric |
|------|--------|
| classification | `accuracy`, `macro_f1`, `precision_macro`, `recall_macro`, `roc_auc`, `pr_auc` |
| regression | `mae`, `rmse`, `r2` |

classification 기준:

- class 균형이 좋으면 accuracy를 볼 수 있습니다.
- class imbalance가 있으면 macro F1, class별 precision/recall을 함께 봅니다.
- positive 탐지가 중요하면 recall을 봅니다.
- 오탐 비용이 크면 precision을 봅니다.
- threshold 조정 가능성이 있으면 ROC-AUC, PR-AUC와 threshold metrics를 확인합니다.

regression 기준:

- 평균 오차 설명은 MAE가 쉽습니다.
- 큰 오차를 강하게 벌점하려면 RMSE를 봅니다.
- R2는 단독으로 결론내지 않습니다.

---

## 6. AutoML

AutoML은 baseline 이후 같은 data_version, split, metric 조건에서 pipeline 후보를 비교하는 단계입니다.
이 템플릿의 기본 backend는 AutoGluon Tabular입니다.

AutoML 전에 학생이 정해야 할 것:

- 실행 목적
- primary metric과 auxiliary metrics
- time_limit
- presets
- leakage 제외 컬럼
- 성공 기준
- test set은 최종 확인용이라는 원칙

AutoGluon 결과는 모델 이름 하나가 아니라 전처리, 모델, 하이퍼파라미터 조합으로 해석합니다.

---

## 7. Model Comparison & Error Analysis

모델 비교 시 반드시 확인합니다.

- 같은 data_version인가?
- 같은 split과 seed인가?
- 같은 metric으로 계산됐는가?
- primary metric과 auxiliary metric이 같은 방향으로 좋아졌는가?
- 성능 차이가 해석 가능한 변경점과 연결되는가?
- 더 복잡한 모델을 쓸 만큼 이득이 충분한가?

오류 분석 순서:

1. confusion matrix 또는 residual 분포 확인
2. 오답 샘플 직접 확인
3. 실패 유형 2~5개로 묶기
4. 각 유형마다 원인 가설과 다음 실험 연결
5. 최종 모델이 어떤 실패 유형을 줄였는지 확인

test set은 최종 확인용입니다. test metric으로 모델 선택이나 threshold 선택을 하지 않습니다.

---

## 8. Demo & Monitoring

Streamlit MVP는 아래를 확인할 수 있어야 합니다.

- Overview: task, target, data_version, split, metric
- Leaderboard: AutoGluon leaderboard 또는 registry 기반 비교
- Evaluation: validation/test metric, confusion matrix
- Prediction: sklearn pickle 또는 AutoGluon predictor 단건 예측
- Logs: `logs/inference.jsonl`, latency, error rate

global feature importance, global SHAP, batch prediction은 MVP 범위에서 제외합니다.
단일 row explanation은 모델 판단의 근거 후보이지 인과 설명으로 쓰지 않습니다.

---

## 9. 실전 요청 예시: 부도 예측

아래 예시는 그대로 복사하기보다 자신의 데이터셋에 맞게 target, positive class, 오류 비용, metric을 바꿔 사용합니다.
좋은 요청은 “무엇을 실행할지”뿐 아니라 “어떤 기준으로 판단할지”를 함께 줍니다.

### Project

```text
/project

부도 예측 tabular classification 프로젝트다.
raw 데이터는 data/raw/bankruptcy.csv, target은 bankrupt로 가정한다.
positive class는 bankrupt=1이고, false negative 비용이 false positive보다 크다.

primary metric은 pr_auc를 우선 검토하되, 실제 선택 기준은
부도 기업 recall을 충분히 확보하면서 precision이 무너지지 않는가로 잡고 싶다.
auxiliary metrics는 recall_macro, precision_macro, macro_f1로 둔다.

reports/PROJECT_REPORT.md에 문제 정의, metric rationale, training_decisions 초안을 정리해줘.
```

### Data

```text
/data data/raw/bankruptcy.csv target=bankrupt

실행 전에 EDA 계획부터 제시해줘.
중점은 class imbalance, missingness pattern, extreme financial ratios,
duplicate company rows, leakage 후보 컬럼이다.

특히 부도 이후에 알 수 있는 사후 정보, 상태값, 정리/상장폐지 관련 컬럼,
ID-like 컬럼, fiscal year/company id가 있는지 확인해줘.
컬럼별로 keep / drop / investigate 결정을 제안하고, 근거를 짧게 써줘.

전처리 실행 전에 reports/PROJECT_REPORT.md의 pipeline_decisions를 업데이트해줘.
```

### Train

```text
/train baseline

processed 데이터 기준으로 baseline을 설계해줘.
부도 예측이므로 class imbalance와 false negative 비용을 우선 고려한다.

실행 전 확인할 것:
- data_version, target, positive_class
- stratified split 가능 여부
- leakage 제외 컬럼이 processed 데이터에서 빠졌는지
- primary_metric=pr_auc가 계산 가능한지
- baseline 모델이 너무 복잡하지 않은지

문제가 없으면 training_decisions를 정리하고 학습 명령을 실행 가능한 상태로 만들어줘.
```

### Evaluate Baseline

```text
/evaluate

baseline 결과를 평가해줘.
pr_auc만 보지 말고 recall_macro, precision_macro, macro_f1, confusion matrix를 같이 봐줘.

특히 bankrupt=1을 놓치는 FN이 어느 정도인지, threshold 조정 여지가 있는지,
metric trade-off가 어떤지 정리해줘.
다음 실험 후보는 데이터 누수 위험이 낮은 것부터 우선순위로 제안해줘.
```

### AutoML

```text
/automl

AutoGluon을 baseline과 같은 data_version, target, split, seed, primary metric 조건으로
비교할 수 있는지 먼저 확인해줘. time_limit은 300초로 시작한다.

채택 기준은 단순히 pr_auc 최고가 아니라:
- bankrupt=1 recall 개선
- precision이 과도하게 붕괴하지 않을 것
- validation/test 차이가 너무 크지 않을 것
- pipeline 복잡도를 감당할 수 있을 것

조건이 맞으면 training_decisions의 AutoML 관련 값을 정리해줘.
```

### Final Evaluation

```text
/evaluate

baseline과 AutoML을 같은 조건에서 비교해줘.
validation metric, test metric, confusion matrix, threshold_metrics, AutoGluon leaderboard를 같이 봐줘.

최종 모델을 하나 고르되, 선택하지 않은 후보의 이유도 남겨줘.
reports/PROJECT_REPORT.md의 Final Decision과 Model Card 내용을 업데이트할 수 있게 정리해줘.
```

### Demo

```text
/demo

Streamlit 데모 기준으로 연결 상태를 점검해줘.
확인 대상은 model_registry, run artifacts, leaderboard, confusion_matrix,
threshold_metrics, prediction, inference log다.

AutoGluon 모델이면 local SHAP explanation이 Prediction 탭에서 작동하는지도 확인 기준에 포함해줘.
단, SHAP은 인과 설명이 아니라 모델 판단 근거 후보로만 해석한다.
```
