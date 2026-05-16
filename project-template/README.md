# Project Title

> 한 줄로 프로젝트를 설명하세요.

---

## 0. A-to-Z Quickstart

이 README는 `project-template/`를 복사하거나 clone한 뒤, 처음 환경 설정부터 AutoGluon AutoML과 Streamlit 확인까지 끝까지 따라가는 가이드입니다.
기본 표준 환경은 **Python 3.10.x**입니다. 이 템플릿은 `.python-version`에 `3.10.13`을 명시합니다.

### 0-1. 프로젝트 폴더로 이동

```bash
cd project-template
```

루트 저장소에서 작업 중이면 `project-template/` 안에서 아래 명령을 실행합니다.

### 0-2. Python 3.10 확인

```bash
python3.10 --version
```

정상 예시는 아래와 같습니다.

```text
Python 3.10.x
```

`python3.10` 명령이 없다면 먼저 Python 3.10을 설치하세요. macOS에서는 `pyenv`, Windows에서는 Python Launcher 또는 공식 설치 파일을 사용할 수 있습니다. 자세한 문제 해결은 `FAQ.md`를 봅니다.

### 0-3. 가상환경 생성과 활성화

macOS / Linux:

```bash
python3.10 -m venv venv
source venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
```

활성화 후 아래 명령이 `venv` 안의 Python을 가리키는지 확인합니다.

```bash
python --version
```

### 0-4. 전체 의존성 설치

이 프로젝트의 의존성 설치는 `requirements.txt` 하나로 통일합니다.
EDA, baseline, AutoGluon, Streamlit 확인에 필요한 패키지를 모두 포함합니다.

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 0-5. 설치 검증

```bash
python --version
python -c "import pandas, sklearn, streamlit; print('core OK')"
python -c "from autogluon.tabular import TabularPredictor; print('autogluon OK')"
python scripts/preprocess.py --help
python scripts/train.py --help
python scripts/train_automl.py --help
```

Streamlit 실행 전 코드 문법만 빠르게 확인하려면 아래도 실행합니다.

```bash
python -m py_compile scripts/preprocess.py scripts/train.py scripts/train_automl.py streamlit_app.py
```

### 0-6. 문제 정의 작성

먼저 `docs/specs/PROJECT_SPEC.md`에 아래 결정을 짧게 적습니다.

- 입력 데이터와 예측 대상
- task type: `classification` 또는 `regression`
- binary classification이면 positive class
- primary metric과 auxiliary metrics
- false positive / false negative / 큰 오차 중 더 중요한 오류
- Streamlit 최종 화면에서 확인할 항목

목표 수치가 없으면 baseline 실행 후 “baseline 대비 개선”으로 성공 기준을 잡습니다.

### 0-7. 원본 데이터 배치

원본 CSV, TSV, TXT 파일을 `data/raw/`에 둡니다.

```text
data/raw/your_raw_file.csv
```

원본 데이터는 Git에 올리지 않습니다. `.gitignore`가 `data/raw/*`를 제외하도록 설정되어 있습니다.

### 0-8. EDA 실행과 Data Card 작성

Claude Code를 사용한다면 아래처럼 요청합니다.

```text
/eda data/raw/[raw_file] target=[target]
```

직접 진행한다면 `notebooks/01_eda_template.ipynb`를 복사해 데이터 이름을 반영한 notebook을 만들고, `docs/specs/DATA_ANALYSIS_SPEC.md` 기준으로 확인합니다.

EDA 후 `reports/DATA_CARD.md`에 반드시 기록합니다.

- row / column 수
- target 분포
- 결측치, 중복, 이상치
- feature redundancy와 drop 후보
- 주요 feature와 target 관계
- 데이터 누수 의심 컬럼
- split 방식과 seed
- processed 파일에 반영할 전처리 결정

### 0-9. 전처리 실행

전처리는 raw 데이터를 학습 가능한 processed CSV로 정돈하는 단계입니다.

```bash
python scripts/preprocess.py \
  --input data/raw/[raw_file] \
  --output data/processed/[processed_file].csv \
  --target [target] \
  --data-version [data_version] \
  --drop-columns [leakage_or_id_columns]
```

제거할 컬럼이 없으면 `--drop-columns` 줄은 빼고 실행합니다.
`--drop-columns`는 EDA와 `reports/DATA_CARD.md`에 근거를 남긴 leakage, ID-like, constant, high-missing, high-correlation 후보를 반영할 때 사용합니다.

header가 없는 파일이면 아래처럼 컬럼명을 지정합니다.

```bash
python scripts/preprocess.py \
  --input data/raw/[raw_file] \
  --output data/processed/[processed_file].csv \
  --target [target] \
  --sep "," \
  --header none \
  --columns col1,col2,col3,target \
  --data-version [data_version]
```

`scaler`, `encoder`, `imputer`처럼 train 데이터에 fit해야 하는 변환은 여기서 하지 않고 학습 pipeline에 남깁니다.

### 0-10. Baseline 학습

classification 예시:

```bash
python scripts/train.py \
  --data data/processed/[processed_file].csv \
  --target [target] \
  --data-version [data_version] \
  --task-type classification \
  --positive-class 1 \
  --primary-metric macro_f1 \
  --metrics accuracy,precision_macro,recall_macro,roc_auc,pr_auc \
  --test-size 0.2 \
  --val-size 0.2
```

regression 예시:

```bash
python scripts/train.py \
  --data data/processed/[processed_file].csv \
  --target [target] \
  --data-version [data_version] \
  --task-type regression \
  --primary-metric rmse \
  --metrics mae,r2 \
  --test-size 0.2 \
  --val-size 0.2
```

학습 후 아래 산출물을 확인합니다.

- `experiments/runs/<run_id>/metrics.json`
- `experiments/runs/<run_id>/test_metrics.json`
- `experiments/runs/<run_id>/predictions.csv`
- `experiments/runs/<run_id>/confusion_matrix.json` classification일 때
- `experiments/runs/<run_id>/threshold_metrics.csv` binary classification이고 probability가 있을 때
- `model_registry.json`

### 0-11. Baseline 결과 검토

Claude Code를 사용한다면 아래 커맨드로 결과를 검토합니다.

```text
/compare-models
/analyze-errors
```

직접 검토할 때는 `metrics.json`, `test_metrics.json`, `predictions.csv`, `confusion_matrix.json`을 확인하고 `reports/EXPERIMENT_REPORT.md`, `reports/ERROR_ANALYSIS.md`에 기록합니다.

중요한 규칙:

- validation metric은 실험 비교 기준입니다.
- test metric은 최종 확인용입니다.
- test set으로 모델 선택이나 threshold 선택을 하지 않습니다.

### 0-12. AutoGluon AutoML 실행

AutoGluon은 baseline 이후 같은 processed CSV, target, split, metric 조건에서 pipeline 후보를 비교하는 단계입니다.
실행 전에 leakage 제외 컬럼이 processed CSV에서 빠졌는지, `data_version`이 baseline과 같은지 확인합니다.

classification 예시:

```bash
python scripts/train_automl.py \
  --data data/processed/[processed_file].csv \
  --target [target] \
  --data-version [data_version] \
  --task-type classification \
  --positive-class 1 \
  --primary-metric macro_f1 \
  --metrics accuracy,precision_macro,recall_macro,roc_auc,pr_auc \
  --test-size 0.2 \
  --val-size 0.2 \
  --time-limit 300 \
  --presets medium_quality
```

regression 예시:

```bash
python scripts/train_automl.py \
  --data data/processed/[processed_file].csv \
  --target [target] \
  --data-version [data_version] \
  --task-type regression \
  --primary-metric rmse \
  --metrics mae,r2 \
  --test-size 0.2 \
  --val-size 0.2 \
  --time-limit 300 \
  --presets medium_quality
```

AutoGluon 실행 후 아래 산출물을 확인합니다.

- `models/automl/<run_id>/`
- `experiments/runs/<run_id>/leaderboard.csv`
- `experiments/runs/<run_id>/automl_summary.json`
- `experiments/runs/<run_id>/metrics.json`
- `experiments/runs/<run_id>/test_metrics.json`
- `model_registry.json`

AutoGluon 결과가 baseline보다 좋아도 validation/test 차이, auxiliary metric 악화, pipeline 복잡도, leakage 가능성을 함께 봅니다.

### 0-13. Streamlit 확인

Claude Code를 사용한다면 아래처럼 요청합니다.

```text
/check-streamlit
```

직접 실행할 때:

```bash
streamlit run streamlit_app.py
```

포트가 이미 사용 중이면:

```bash
streamlit run streamlit_app.py --server.port 8502
```

Streamlit에서 최소한 아래를 확인합니다.

- Overview: task, target, data_version, split, metric
- Leaderboard: AutoGluon leaderboard 또는 registry 기반 실험 비교
- Evaluation: validation/test metric, confusion matrix
- Prediction: sklearn pickle 또는 AutoGluon predictor 단건 예측
- Logs: `logs/inference.jsonl`, latency, error rate

### 0-14. Checkpoint

단계가 끝날 때마다 문서와 가벼운 기록 파일만 commit합니다.

```text
/checkpoint [메시지]
```

Git에 올리지 않는 항목:

- `data/raw/*`
- `data/processed/*`
- `models/*`
- `experiments/runs/*`
- `logs/*`
- `.env`
- `.streamlit/secrets.toml`

---

## 1. 프로젝트 개요

- **문제 정의**:
- **입력 -> 출력**:
- **사용자 / 활용 상황**:
- **Task type / positive class**:
- **주요 평가 지표**:

문제 정의와 성공 기준은 `docs/specs/PROJECT_SPEC.md`에 먼저 정리합니다.
Claude Code와 협업할 때의 작업 기준은 `CLAUDE.md`를 따릅니다.

---

## 2. 데이터 분석

- **데이터 출처**:
- **데이터 버전**: `data_manifest.json`의 `data_version`
- **데이터 크기**:
- **주요 컬럼 / 입력**:
- **EDA 핵심 발견**:
- **품질 이슈 / 편향 가능성**:
- **데이터 누수 점검 결과**:

자세한 내용은 `reports/DATA_CARD.md`에 기록합니다.
분석 기준은 `docs/specs/DATA_ANALYSIS_SPEC.md`를 따릅니다.

---

## 3. 전처리

전처리 단계에서는 raw 데이터를 모델 학습용 processed CSV로 정돈합니다.

- **raw 파일 경로**:
- **processed 파일 경로**:
- **target column**:
- **적용한 전처리**:
- **제거한 컬럼과 이유**:
- **train pipeline에 남긴 fit 기반 변환**:

컬럼명 정리, rename/drop, target mapping, 중복 제거, target 결측 제거처럼 파일 자체를 정돈하는 작업은 `scripts/preprocess.py`에서 처리합니다.
scaler, encoder, imputer처럼 train 데이터에 fit해야 하는 변환은 학습 pipeline에 남깁니다.

---

## 4. 모델링과 실험

- **Baseline 모델**:
- **평가 지표 선택 이유**:
- **AutoML 사용 여부와 목적**:
- **주요 실험 요약**:
- **최종 모델 선택 이유**:

| model_id | data_version | 주요 변경점 | metric | 한계 |
|----------|--------------|-------------|--------|------|
|          |              |             |        |      |

자세한 실험 로그는 `experiments/runs/`와 `model_registry.json`을 확인합니다.
Streamlit에서 볼 confusion matrix는 각 run의 `confusion_matrix.json`을 사용합니다.
AutoGluon을 실행했다면 `leaderboard.csv`와 `automl_summary.json`을 함께 확인합니다.
모델링 기준은 `docs/specs/MODELING_SPEC.md`를 따릅니다.

---

## 5. 결과 검토와 재모델링

- **오류 유형**:
- **개선한 부분**:
- **개선되지 않은 부분**:
- **다음에 시도할 것**:

자세한 내용은 `reports/ERROR_ANALYSIS.md`와 `reports/EXPERIMENT_REPORT.md`에 기록합니다.
metric 해석과 모델 비교 기준은 `docs/specs/METRICS_AND_INTERPRETATION_SPEC.md`를 따릅니다.

---

## 6. 버전 관리

- **데이터 버전 기록**: `data_manifest.json`
- **모델 버전 기록**: `model_registry.json`
- **실험 산출물**: `experiments/runs/<run_id>/`
- **큰 파일 보관 위치**: [Google Drive / Hugging Face Hub / Kaggle / 기타 링크]

큰 원본 데이터, 가공 데이터, 모델 가중치는 Git에 직접 올리지 않습니다.
작업 단계가 끝나면 `/checkpoint [메시지]`로 `*.md`, 코드, `data_manifest.json`, `model_registry.json` 같은 가벼운 기록 파일만 검토 후 commit합니다.
Claude Code와 협업할 때의 공통 기준은 `CLAUDE.md`를 확인합니다.

---

## 7. Streamlit 시각화와 모니터링

- **데모 방식**: Streamlit Community Cloud / 로컬 실행 / 기타
- **데모 URL**:
- **기록하는 로그**: `logs/inference.jsonl`
- **실험 대시보드**: Overview, Leaderboard, Evaluation, Prediction, Logs
- **AutoML 대시보드**: `leaderboard.csv`, `automl_summary.json`, validation/test metric
- **Threshold 대시보드**: binary classification에서 `threshold_metrics.csv`가 있을 때 precision/recall/F1 확인
- **Local explanation**: AutoGluon 내부 모델 2개에 같은 입력 row를 요청하고 SHAP으로 응답 이유 후보 비교
- **로그 대시보드**: 요청 수, error rate, 평균/P95 latency, 최근 추론 로그
- **운영 한계**:

---

## 8. 결과 및 한계

- **최종 결과**:
- **잘 된 점**:
- **한계**:
- **개선 방향**:

---

## 9. 자연어로 추가 검토 요청하기

정해진 산출물을 만들 때는 slash command를 우선 사용합니다.
판단과 해석이 필요한 질문은 아래처럼 자연어로 요청하면 좋습니다.

```markdown
이 데이터셋의 컬럼 설명, 결측치/이상치 요약, target 분포를 바탕으로
추가로 확인해야 할 EDA 질문 10개를 제안해주세요.
데이터 누수 가능성이 있는 컬럼도 따로 의심 목록으로 정리해주세요.
```

```markdown
현재 문제 정의와 EDA 결과를 바탕으로 feature engineering 후보를 제안해주세요.
각 feature마다 기대 효과, 데이터 누수 위험, 구현 난이도를 표로 정리해주세요.
```

```markdown
아래 실험 결과에서 metric이 실제 문제 정의와 잘 맞는지 검토해주세요.
accuracy만 볼 때 놓칠 수 있는 위험과 추가로 봐야 할 지표를 제안해주세요.
```

```markdown
오답 샘플과 예측 확률을 보고 실패 유형을 묶어주세요.
각 유형별 원인 가설과 다음 실험 아이디어를 제안해주세요.
```

AI가 만든 해석은 실제 데이터 분포, split, metric, 중복/누수 계산으로 다시 확인합니다.
