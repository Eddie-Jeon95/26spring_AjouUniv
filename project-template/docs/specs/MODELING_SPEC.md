# Modeling Spec

모델링의 목표는 가장 복잡한 모델을 쓰는 것이 아니라, 가설을 세우고 검증하면서 모델을 설명 가능하게 개선하는 것입니다.

---

## 1. Baseline 원칙

baseline은 다음 조건을 만족해야 합니다.

- 단순하고 빠르게 실행된다.
- 재현 가능한 config와 seed를 가진다.
- 주요 metric과 보조 metric을 기록한다.
- 이후 실험의 비교 기준이 된다.

권장 baseline:

| 문제 유형 | baseline 예시 |
|-----------|---------------|
| classification | Logistic Regression, RandomForest, DummyClassifier |
| regression | Linear Regression, RandomForestRegressor, DummyRegressor |
| text | TF-IDF + linear model |
| image | pretrained feature extractor 또는 작은 CNN |

---

## 2. 실험 가설

새 실험은 아래 문장을 채울 수 있어야 합니다.

```markdown
[변경점]을 적용하면 [현재 관찰한 문제]가 줄어들 것이고,
그 결과 [metric 또는 error type]이 개선될 것이다.
```

좋은 실험 예시:

- class imbalance가 있으므로 class weight를 적용해 minority recall을 개선한다.
- 특정 수치 feature의 skew가 크므로 log transform을 적용해 linear baseline을 개선한다.
- 오답이 짧은 텍스트에 몰려 있으므로 text length feature를 추가한다.

나쁜 실험 예시:

- 성능이 오를 것 같아서 큰 모델로 바꾼다.
- 이유 없이 parameter를 여러 개 동시에 바꾼다.
- split이 다른 결과를 같은 baseline과 비교한다.

---

## 3. AutoGluon AutoML 원칙

AutoML은 baseline 이후에 사용하는 pipeline 비교 실험입니다.
이 템플릿의 기본 AutoML backend는 **AutoGluon Tabular만** 사용합니다.

AutoGluon 실행 전에 `/plan-automl`에서 아래를 고정합니다.

- task type과 target column
- positive class
- primary metric과 auxiliary metrics
- `data_version`, split 비율, seed
- leakage 제외 컬럼과 processed CSV
- 성공 기준과 운영상 trade-off

AutoML은 계획을 대신 세우지 않습니다. AutoGluon은 전처리, 모델, 하이퍼파라미터 조합을 탐색하지만, 어떤 metric을 최적화할지와 어떤 오류를 더 중요하게 볼지는 사람이 정합니다.

실행 기준:

- `scripts/train_automl.py`를 사용합니다.
- baseline과 같은 processed CSV, target, split, seed, metric 조건에서 실행합니다.
- AutoGluon에는 train과 validation만 전달합니다.
- test set은 `evaluate()`와 `leaderboard(test_data)`로 최종 확인할 때만 사용합니다.
- AutoGluon artifact는 `models/automl/<run_id>/`에 저장합니다.
- run directory에는 `leaderboard.csv`, `automl_summary.json`, `metrics.json`, `test_metrics.json`을 저장합니다.
- binary classification에서 positive class와 probability가 있으면 `threshold_metrics.csv`를 저장합니다.

AutoGluon 결과가 baseline보다 좋아도 다음을 함께 봅니다.

- validation/test 성능 차이
- primary metric과 auxiliary metric의 trade-off
- fit time과 추론 가능성
- pipeline 복잡도
- leakage 후보 feature 사용 여부
- test set이 모델 선택에 쓰이지 않았는지

global feature importance, global SHAP, batch prediction은 현재 MVP 범위에서 제외합니다.
단, AutoGluon 내부 모델 2개를 같은 입력 row로 비교하는 local SHAP explanation은 Prediction 탭에서 지원합니다.
이 설명은 모델 판단의 근거 후보이지 인과 설명이 아닙니다.

---

## 4. Config 관리

`configs/default.yaml`은 항상 실행 가능한 demo/default 예시여야 합니다.
실제 프로젝트에서는 학생이 YAML을 직접 수정하기보다 `scripts/train.py`에 CLI 인자를 넘겨 실행합니다.
Data Card에서 정한 split을 맞출 때도 YAML을 고치기보다 `--test-size`, `--val-size`, `--no-stratify` 인자를 우선 사용합니다.

실험이 늘어나면 다음 중 하나로 관리합니다.

- 작은 프로젝트: `python scripts/train.py --data ... --target ... --data-version ... --test-size ... --val-size ...`로 실행
- 기록 중심: 실행된 effective config가 `experiments/runs/<run_id>/config.yaml`에 저장되는 것을 활용
- 심화 프로젝트: 필요하면 `configs/default.yaml`을 복사해 `configs/<experiment>.yaml` 생성
- 심화 프로젝트: MLflow나 DVC를 선택적으로 추가

config에는 최소한 seed, task type, positive class, data path, target column, data_version, model name, model params, primary metric, auxiliary metrics가 있어야 합니다.

---

## 5. 실험 기록 기준

학습 실행 후 다음이 남아야 합니다.

- `experiments/runs/<run_id>/config.yaml`
- `experiments/runs/<run_id>/metrics.json`
- `experiments/runs/<run_id>/predictions.csv`
- `experiments/runs/<run_id>/confusion_matrix.json`
- `experiments/runs/<run_id>/result.md`
- `model_registry.json`의 새 record

기록할 핵심 필드:

- model_id
- run_id
- data_version
- config_hash
- primary_metric
- auxiliary_metrics
- task_type
- positive_class
- metrics
- artifact_path
- confusion_matrix_path
- threshold_metrics_path
- limitations

AutoGluon 실험 record에는 추가로 아래 필드를 기록합니다.

- `experiment_type=automl`
- `backend=autogluon`
- `leaderboard_path`
- `automl_summary_path`

---

## 6. 모델 개선 기준

모델이 개선됐다고 말하려면 다음 중 2개 이상을 제시하세요.

- 주요 metric 개선
- 문제 비용과 관련된 보조 metric 개선
- 특정 error type 감소
- 더 단순하거나 빠른 모델로 유사 성능 달성
- 데이터 누수 가능성을 줄이면서 성능 유지
- 해석 가능성 또는 재현성 개선

단순히 test accuracy가 조금 올랐다는 이유만으로 최종 모델을 선택하지 않습니다.

---

## 7. 최종 모델 선택 기준

최종 모델은 다음 질문에 답할 수 있어야 합니다.

- 어떤 baseline보다 왜 나은가?
- 어떤 데이터 버전에서 학습됐는가?
- 어떤 입력에서 약한가?
- 데모 환경에서 실행 가능한가?
- 모델 파일은 어디에 있고 재현 가능한가?
- `reports/MODEL_CARD.md`에 한계가 기록됐는가?
