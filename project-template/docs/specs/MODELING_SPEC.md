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

## 3. Config 관리

`configs/default.yaml`은 항상 실행 가능한 기본값이어야 합니다.

실험이 늘어나면 다음 중 하나로 관리합니다.

- 작은 프로젝트: `configs/default.yaml`을 복사해 `configs/<experiment>.yaml` 생성
- 기록 중심: 실행된 config가 `experiments/runs/<run_id>/config.yaml`에 복사되는 것을 활용
- 심화 팀: MLflow나 DVC를 선택적으로 추가

config에는 최소한 seed, data path, target column, data_version, model name, model params, primary metric이 있어야 합니다.

---

## 4. 실험 기록 기준

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
- metrics
- artifact_path
- confusion_matrix_path
- limitations

---

## 5. 모델 개선 기준

모델이 개선됐다고 말하려면 다음 중 2개 이상을 제시하세요.

- 주요 metric 개선
- 문제 비용과 관련된 보조 metric 개선
- 특정 error type 감소
- 더 단순하거나 빠른 모델로 유사 성능 달성
- 데이터 누수 가능성을 줄이면서 성능 유지
- 해석 가능성 또는 재현성 개선

단순히 test accuracy가 조금 올랐다는 이유만으로 최종 모델을 선택하지 않습니다.

---

## 6. 최종 모델 선택 기준

최종 모델은 다음 질문에 답할 수 있어야 합니다.

- 어떤 baseline보다 왜 나은가?
- 어떤 데이터 버전에서 학습됐는가?
- 어떤 입력에서 약한가?
- 데모 환경에서 실행 가능한가?
- 모델 파일은 어디에 있고 재현 가능한가?
- `reports/MODEL_CARD.md`에 한계가 기록됐는가?
