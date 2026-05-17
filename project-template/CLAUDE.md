# Claude Code Guide

이 문서는 Claude Code가 이 프로젝트에서 따라야 할 단일 작업 기준입니다.
프로젝트 판단 기준은 `PROJECT_GUIDE.md`, 기록과 실행 결정은 `reports/PROJECT_REPORT.md`를 따릅니다.

---

## 1. 목적

이 템플릿의 목표는 학생들이 ML 프로젝트를 다음 흐름으로 끝까지 수행하도록 돕는 것입니다.

1. 문제 정의
2. 데이터 분석과 품질 점검
3. 전처리 결정과 processed CSV 생성
4. baseline 모델 구축
5. AutoML 필요성 판단과 후보 비교
6. 모델 비교와 error analysis
7. Streamlit 기반 데모 확인
8. 버전 기록과 checkpoint

코드를 대신 완성하는 것보다, 학생이 중요한 결정을 확인하고 근거를 남기게 하는 것이 우선입니다.

---

## 2. 먼저 읽을 문서

작업 시작 시 다음 순서로 컨텍스트를 잡습니다.

1. `CLAUDE.md`
2. `README.md`
3. `PROJECT_GUIDE.md`
4. `reports/PROJECT_REPORT.md`
5. `data_manifest.json`, `model_registry.json`

작업 요청이 애매하면 위 문서 기준으로 부족한 정보를 확인하고, 추측으로 코드를 확장하지 않습니다.

---

## 3. 폴더와 산출물

```text
configs/              # 실행 가능한 demo/default YAML
data/raw/             # 원본 데이터, Git 제외
data/processed/       # 가공 데이터, Git 제외
experiments/runs/     # 실행별 metric, prediction, config, Git 제외
logs/                 # inference log, Git 제외
models/               # model artifact, Git 제외
notebooks/            # EDA notebook
reports/PROJECT_REPORT.md # Data Card, Experiment Report, Error Analysis, Model Card
scripts/preprocess.py # raw -> processed
scripts/train.py      # baseline/custom experiment
scripts/train_automl.py # AutoGluon Tabular
streamlit_app.py      # 실험 결과와 추론 로그 대시보드
```

핵심 기록:

- `reports/PROJECT_REPORT.md`: 학생 판단, decision block, 최종 보고서
- `data_manifest.json`: data_version, source, checksum, split
- `model_registry.json`: model_id, metric, artifact path
- `experiments/runs/<run_id>/`: 실행 config와 metric 산출물

---

## 4. 권장 Slash Commands

| 커맨드 | 역할 |
|--------|------|
| `/project` | 문제 정의, target, task type, metric, 성공 기준 |
| `/data` | EDA, drop 결정, `pipeline_decisions`, preprocessing |
| `/train` | baseline/일반 실험, `training_decisions`, 학습 실행 |
| `/automl` | AutoGluon 실행 조건 고정과 결과 기록 |
| `/evaluate` | 모델 비교, error analysis, 최종 모델 선택 |
| `/demo` | Streamlit, prediction, inference log 확인 |
| `/checkpoint` | 변경 파일 검토와 git checkpoint |

정해진 산출물과 실행 흐름은 slash command를 사용합니다.
판단 상담, metric 설명, 컬럼별 trade-off 검토는 자연어 채팅으로 처리합니다.

---

## 5. 작업 원칙

### 데이터

- 원본 데이터와 processed 데이터는 Git에 올리지 않습니다.
- raw 파일이 없으면 중단하고, 예제 데이터를 자동 생성하거나 다운로드하지 않습니다.
- EDA는 바로 실행하지 말고 먼저 확인 질문과 표/그래프 계획을 짧게 제시합니다.
- drop 후보는 `keep / drop / investigate` 결정으로 학생에게 확인합니다.
- leakage가 의심되면 성능 개선보다 누수 확인을 우선합니다.
- 전처리 결정은 `reports/PROJECT_REPORT.md`의 `pipeline_decisions` block에 기록합니다.
- scaler, encoder, imputer처럼 fit이 필요한 변환은 processed CSV에 미리 반영하지 않습니다.

### 모델링

- 복잡한 모델보다 단순한 baseline을 먼저 만듭니다.
- task type, positive class, metric, split, baseline 모델은 학생이 확인해야 합니다.
- 실행 조건은 `training_decisions` block에 기록합니다.
- 기본 학습 명령은 `python scripts/train.py --decisions reports/PROJECT_REPORT.md`입니다.
- 새 실험은 가설, 변경점, 비교 baseline, 성공/실패 기준을 가져야 합니다.
- 실행에 사용된 effective config는 run 폴더의 `config.yaml`에 자동 저장됩니다.

### AutoML

- AutoML은 baseline 이후 사용합니다.
- baseline과 같은 processed CSV, target, data_version, split, seed, metric 조건에서만 직접 비교합니다.
- AutoGluon에는 train과 validation만 전달하고 test set은 최종 확인에만 사용합니다.
- AutoML 결과는 전처리, 모델, 하이퍼파라미터 pipeline 후보 비교로 해석합니다.

### 결과 해석

- metric 하나만 보고 결론내지 않습니다.
- validation metric은 실험 비교 기준, test metric은 최종 확인용입니다.
- data_version이나 split이 다르면 직접 우열을 말하지 않습니다.
- error analysis는 실패 유형, 원인 가설, 다음 실험으로 연결합니다.
- 최종 모델 한계는 `PROJECT_REPORT.md`의 Model Card 섹션에 기록합니다.

### Claude 사용

- API key, 개인정보, 민감한 원본 샘플은 프롬프트에 넣지 않습니다.
- 새 라이브러리는 꼭 필요한 경우에만 제안하고 `requirements.txt`를 함께 갱신합니다.
- 이미 requirements에 있는 패키지는 `pip install -r requirements.txt`로 설치하도록 안내합니다.
- 결과가 좋아 보여도 leakage, split 변경, metric 선택 문제를 먼저 점검합니다.

---

## 6. 금지 사항

- 데이터 파일, 모델 가중치, `.env`, Streamlit secrets를 Git에 추가하지 않습니다.
- test set 성능을 반복적으로 보고 모델이나 threshold를 선택하지 않습니다.
- 실패한 실험을 삭제하거나 숨기지 않습니다.
- metric 개선 근거 없이 "성능 향상"이라고 쓰지 않습니다.
- `PROJECT_REPORT.md`, `data_manifest.json`, `model_registry.json`을 갱신하지 않은 채 최종 모델만 남기지 않습니다.
