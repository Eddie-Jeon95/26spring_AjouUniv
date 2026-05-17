# Project Title

> 한 줄로 프로젝트를 설명하세요.

이 템플릿은 ML 프로젝트를 문제 정의, 데이터 분석, 전처리, baseline, AutoML, 평가, Streamlit 데모까지 한 흐름으로 진행하도록 돕습니다.
긴 Python 명령어를 외우기보다 `reports/PROJECT_REPORT.md`의 decision block을 채우고, slash command로 단계별 진행을 요청하는 방식을 권장합니다.

---

## 0. Quickstart

### 0-1. 환경 준비

```bash
cd project-template
python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell에서는 아래처럼 가상환경을 만듭니다.

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
```

설치 확인:

```bash
python --version
python -c "import pandas, sklearn, streamlit; print('core OK')"
python -c "from autogluon.tabular import TabularPredictor; print('autogluon OK')"
```

### 0-2. 원본 데이터 배치

원본 CSV, TSV, TXT 파일을 `data/raw/`에 둡니다.

```text
data/raw/your_raw_file.csv
```

원본 데이터, processed 데이터, 모델 파일, 실행 산출물, 로그는 Git에 올리지 않습니다.

### 0-3. 권장 진행 순서

Claude Code를 사용한다면 아래 단계로 진행합니다.

```text
/project
/data data/raw/[raw_file] target=[target]
/train baseline
/automl
/evaluate
/demo
```

직접 명령을 실행할 때도 기본은 decision block 기반입니다.

```bash
python scripts/preprocess.py --decisions reports/PROJECT_REPORT.md
python scripts/train.py --decisions reports/PROJECT_REPORT.md
python scripts/train_automl.py --decisions reports/PROJECT_REPORT.md
```

CLI 인자는 임시 override가 필요할 때만 사용합니다.

---

## 1. 핵심 문서

| 파일 | 역할 |
|------|------|
| `README.md` | 학생용 시작 가이드 |
| `CLAUDE.md` | Claude Code 작업 원칙 |
| `PROJECT_GUIDE.md` | 문제 정의, EDA, 모델링, metric, AutoML, 평가 기준 |
| `reports/PROJECT_REPORT.md` | Data Card, Experiment Report, Error Analysis, Model Card, decision block |
| `FAQ.md` | 설치와 자주 막히는 문제 |

`PROJECT_REPORT.md` 안의 YAML block은 실행 스크립트가 직접 읽습니다.
자유 서술과 표는 기록용이고, 실제 실행 설정은 `pipeline_decisions`, `training_decisions`만 사용합니다.

---

## 2. 단계별 흐름

### `/project`: 문제 정의

학생이 직접 확인할 결정:

- 입력 데이터와 target
- task type과 positive class
- 중요한 오류 비용
- primary metric과 auxiliary metrics
- Streamlit 최종 화면에서 확인할 항목

기록 위치: `reports/PROJECT_REPORT.md`의 Project Definition 섹션

### `/data`: EDA와 전처리

확인할 것:

- target 분포
- 결측치, 중복, 이상치
- feature redundancy
- ID-like, high-missing, constant, high-correlation 후보
- leakage 의심 컬럼
- split 방식

학생이 `keep / drop / investigate` 결정을 확인한 뒤 `pipeline_decisions`를 채우고 전처리를 실행합니다.

```bash
python scripts/preprocess.py --decisions reports/PROJECT_REPORT.md
```

### `/train`: baseline과 일반 실험

확인할 것:

- baseline 모델
- primary/auxiliary metrics
- split과 seed
- feature 정책
- 새 실험이면 가설과 비교 기준

`training_decisions`를 채운 뒤 실행합니다.

```bash
python scripts/train.py --decisions reports/PROJECT_REPORT.md
```

### `/automl`: AutoGluon AutoML

AutoML은 baseline 이후 같은 data_version, split, metric 조건에서 pipeline 후보를 비교하는 단계입니다.

학생이 확인할 것:

- 실행 목적
- time_limit과 presets
- 성공 기준
- auxiliary metric 악화 허용 여부
- test set은 최종 확인용이라는 원칙

```bash
python scripts/train_automl.py --decisions reports/PROJECT_REPORT.md
```

### `/evaluate`: 모델 비교와 오류 분석

확인할 것:

- 같은 data_version, split, metric인지
- validation/test 차이
- confusion matrix 또는 residual
- threshold 조정 필요성
- 실패 유형과 다음 실험
- 최종 모델 선택 이유

기록 위치: Error Analysis, Model Card, Final Checklist 섹션

### `/demo`: Streamlit 데모

직접 실행:

```bash
streamlit run streamlit_app.py
```

확인할 탭:

- Overview
- Leaderboard
- Evaluation
- Prediction
- Logs

## 3. 주요 산출물

- `data_manifest.json`: data_version, row/column 수, checksum, split 기록
- `model_registry.json`: model_id, metric, artifact path, run path 기록
- `experiments/runs/<run_id>/config.yaml`: 실행에 사용된 effective config
- `experiments/runs/<run_id>/metrics.json`: validation metric
- `experiments/runs/<run_id>/test_metrics.json`: 최종 확인용 test metric
- `experiments/runs/<run_id>/predictions.csv`: error analysis용 예측 샘플
- `experiments/runs/<run_id>/confusion_matrix.json`: Streamlit Evaluation 탭
- `experiments/runs/<run_id>/leaderboard.csv`: AutoGluon 후보 비교
- `logs/inference.jsonl`: Streamlit 추론 로그

---

## 4. Git 제외 항목

아래 항목은 Git에 추가하지 않습니다.

- `data/raw/*`
- `data/processed/*`
- `models/*`
- `experiments/runs/*`
- `logs/*`
- `.env`
- `.streamlit/secrets.toml`

---

## 5. 자연어로 물어보기 좋은 것

정해진 단계는 slash command로 진행하고, 판단이 필요한 부분은 자연어로 물어보면 좋습니다.

```markdown
이 drop 후보 중 어떤 컬럼을 유지하거나 제거하는 게 좋을지,
데이터 누수 위험과 모델 해석 관점에서 비교해주세요.
```

```markdown
현재 문제 정의에서 accuracy와 macro_f1 중 어떤 metric이 더 적절한지 설명해주세요.
```

```markdown
AutoML 결과가 baseline보다 좋아 보이는데, 실제로 채택해도 되는지
validation/test 차이와 auxiliary metric 기준으로 검토해주세요.
```
