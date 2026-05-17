# Project Report

이 파일은 프로젝트의 판단과 실행 결정을 한곳에 남기는 보고서입니다.
아래 YAML decision block은 실행 스크립트가 직접 읽습니다. 자유 서술과 표는 기록용이고, 실제 실행 설정은 decision block만 사용합니다.

---

## 1. Project Definition

- **문제 정의**:
- **입력 데이터**:
- **target column**:
- **task type / positive class**:
- **사용자와 사용 상황**:
- **중요한 오류**:
- **primary metric / auxiliary metrics**:
- **성공 기준**:
- **Streamlit 확인 항목**:

---

## 2. Data Card

- **데이터 이름**:
- **데이터 버전**:
- **출처 / 링크**:
- **수집 기간 / 조건**:
- **라이선스 / 사용 조건**:
- **raw 파일 경로**:
- **processed 파일 경로**:
- **실행한 EDA notebook**:

### 구조와 품질

- **row 수**:
- **column 수**:
- **target 분포**:
- **결측치 / 이상치 / 중복**:
- **주요 feature**:
- **feature redundancy**:
- **데이터 누수 의심 컬럼**:
- **split 방식 추천**:
- **편향 / 개인정보 / 라벨 품질 위험**:

### Feature Decisions

| column | issue | evidence | AI recommendation | student decision |
|--------|-------|----------|-------------------|------------------|
|        |       |          |                   |                  |

### Pipeline Decisions

```yaml pipeline_decisions
input: ""
output: ""
target: ""
data_version: ""
sep: ","
header: infer
columns: []
drop_columns: []
rename: {}
target_map: {}
drop_duplicate_rows: false
keep_rows_missing_target: false
manifest: data_manifest.json
source: ""
notes: ""
```

---

## 3. Experiment Report

### Baseline / Experiment Plan

- **사용할 processed 데이터**:
- **data_version**:
- **target column**:
- **task type / positive class**:
- **feature 정책**:
- **제외할 feature와 이유**:
- **split 방식 / seed**:
- **primary metric / auxiliary metrics**:
- **baseline 또는 실험 모델**:
- **선택 이유**:
- **실험 가설**:
- **train 전 누수 위험 점검**:
- **실행 명령**:

```bash
python scripts/train.py --decisions reports/PROJECT_REPORT.md
```

### Training Decisions

```yaml training_decisions
data: ""
target: ""
data_version: ""
task_type: classification
positive_class: null
primary_metric: accuracy
metrics:
  - macro_f1
  - precision_macro
  - recall_macro
split:
  test_size: 0.2
  val_size: 0.1
  stratify: true
baseline:
  experiment_name: baseline
  model_name: logistic_regression
  model_params:
    max_iter: 1000
automl:
  time_limit: 300
  presets: medium_quality
```

### Baseline / Experiment Result

- **model_id**:
- **run_id**:
- **validation 주요 metric**:
- **test 주요 metric**:
- **보조 metric trade-off**:
- **threshold 필요 여부**:
- **비교 기준으로 삼을 이유**:
- **한계**:

### AutoML Plan & Result

- **AutoML backend**: AutoGluon Tabular
- **실행 목적**:
- **고정할 split / seed / metric**:
- **leakage 제외 컬럼**:
- **time_limit / presets**:
- **성공 기준**:
- **selected AutoGluon model**:
- **leaderboard_path**:
- **automl_summary_path**:
- **baseline 대비 개선/악화**:
- **채택 또는 보류 이유**:

### 실험 요약

| run_id | type | 가설/목적 | 변경점 | 주요 metric | 결과 해석 | 다음 액션 |
|--------|------|-----------|--------|-------------|-----------|-----------|
|        |      |           |        |             |           |           |

---

## 4. Error Analysis

- **분석한 run_id / model_id**:
- **사용한 파일**: `predictions.csv`, `confusion_matrix.json`, `threshold_metrics.csv`
- **주요 오류 유형**:

| 실패 유형 | 근거 | 원인 가설 | 다음 실험 |
|-----------|------|-----------|-----------|
|           |      |           |           |

- **개선된 오류 유형**:
- **아직 남은 오류 유형**:
- **test set을 모델 선택에 쓰지 않았는지**:

---

## 5. Model Card

- **최종 model_id**:
- **experiment type**: baseline / automl / custom
- **data_version**:
- **artifact path**:
- **사용한 feature 정책**:
- **primary metric 선택 이유**:
- **validation/test 주요 결과**:
- **강한 입력 조건**:
- **약한 입력 조건**:
- **사용하면 안 되는 상황**:
- **운영 위험과 모니터링 항목**:
- **재현 명령**:

```bash
python scripts/train.py --decisions reports/PROJECT_REPORT.md
python scripts/train_automl.py --decisions reports/PROJECT_REPORT.md
```

---

## 6. Demo & Monitoring

- **데모 방식**: 로컬 Streamlit / Streamlit Cloud / 기타
- **데모 URL**:
- **확인한 탭**: Overview / Leaderboard / Evaluation / Prediction / Logs
- **추론 로그 위치**: `logs/inference.jsonl`
- **latency / error rate 확인 결과**:
- **데모에서 보여줄 모델**:
- **데모 한계**:

---

## 7. Final Checklist

- [ ] 문제 정의, target, task type, metric을 설명할 수 있다.
- [ ] EDA에서 결측치, 중복, 이상치, target 분포를 확인했다.
- [ ] drop 후보와 최종 결정을 근거와 함께 남겼다.
- [ ] `pipeline_decisions`로 processed CSV를 만들었다.
- [ ] `data_manifest.json`에 data_version과 checksum이 남았다.
- [ ] baseline을 먼저 만들었다.
- [ ] 새 실험마다 가설과 비교 기준을 남겼다.
- [ ] baseline과 AutoML을 같은 data_version, split, metric 기준에서 비교했다.
- [ ] test set을 모델 선택이나 threshold 선택에 사용하지 않았다.
- [ ] error analysis가 다음 실험 또는 최종 선택 근거와 연결된다.
- [ ] 최종 모델의 한계와 사용 금지 조건을 Model Card에 기록했다.
- [ ] Streamlit에서 모델/run/log를 확인했다.
