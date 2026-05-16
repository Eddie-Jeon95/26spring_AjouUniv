# Claude Code Guide

> 이 문서는 Claude Code가 이 프로젝트에서 따라야 할 단일 작업 기준입니다.
> 프로젝트 판단 기준은 이 문서와 `docs/specs/`를 함께 따릅니다.

---

## 1. 프로젝트 목적

이 템플릿의 목표는 학생들이 ML 프로젝트를 다음 흐름으로 끝까지 수행하도록 돕는 것입니다.

1. 문제 정의
2. 데이터 분석과 품질 점검
3. baseline 모델 구축
4. baseline 결과 분석과 AutoML 필요성 판단
5. AutoGluon 기반 pipeline 후보 비교
6. 모델/데이터 버전 기록
7. 결과 해석과 error analysis
8. Streamlit 기반 실험 결과와 추론 로그 시각화

코드를 대신 완성하는 것보다, 학생이 판단 근거를 남기며 개선하도록 돕는 것이 우선입니다.

---

## 2. 먼저 읽을 문서

Claude Code로 작업을 시작할 때는 다음 순서로 컨텍스트를 잡습니다.

1. `CLAUDE.md`: Claude Code 작업 기준과 폴더 규칙
2. `project-template/README.md`: 학생이 따라갈 전체 진행 순서
3. `docs/specs/`: 문제 정의, 데이터 분석, 모델링, 지표 해석 기준
4. `reports/`: 데이터 카드, 실험 리포트, 오류 분석, 모델 카드
5. `MLOPS_CHECKLIST.md`: 진행 중 자가 점검

작업 요청이 애매하면 위 spec을 기준으로 부족한 정보를 먼저 확인하고, 추측으로 코드를 확장하지 않습니다.

---

## 3. 환경 Preflight

clone 직후 또는 새 세션에서 실제 프로젝트를 시작할 때는 코드 작업보다 먼저 환경을 확인합니다.

```bash
python --version
python -c "import pandas, sklearn, streamlit; print('core OK')"
python -c "from autogluon.tabular import TabularPredictor; print('autogluon OK')"
python scripts/preprocess.py --help
python scripts/train.py --help
python scripts/train_automl.py --help
```

기준:

- Python은 `3.10.x`를 사용합니다. `.python-version`의 기본값은 `3.10.13`입니다.
- A-to-Z 기본 설치는 `pip install -r requirements-automl.txt`입니다.
- 가벼운 baseline 확인만 할 때만 `pip install -r requirements.txt`를 보조 선택지로 안내합니다.
- 패키지를 하나씩 최신 버전으로 설치하지 않습니다. 고정된 requirements 파일을 기준으로 재현성을 맞춥니다.
- AutoGluon import가 실패하면 AutoML 실행 전에 환경 문제를 먼저 해결합니다.

---

## 4. 폴더와 산출물 규칙

```
configs/              # 기본 학습 설정 YAML (학생이 직접 수정하는 문서가 아님)
data/raw/             # 원본 데이터, Git 제외
data/processed/       # 가공 데이터, Git 제외
docs/specs/           # 프로젝트 판단 기준 문서
experiments/runs/     # 실행별 config, metrics, predictions, confusion matrix, result
notebooks/            # EDA와 탐색 분석
reports/              # 데이터/실험/오류/모델 해석 리포트
scripts/preprocess.py # raw 데이터를 processed CSV로 정돈하는 진입점
scripts/train.py      # baseline 학습과 실험 기록 진입점
scripts/train_automl.py # AutoGluon Tabular 학습과 실험 기록 진입점
src/data/             # 전처리와 데이터 로드 확장 지점
src/models/           # 모델 정의와 학습/추론 확장 지점
src/utils/            # 로깅 등 공통 유틸리티
streamlit_app.py      # 실험 결과와 추론 로그를 보여주는 앱 진입점
logs/                 # 추론 로그, Git 제외
```

작업 단계별로 주로 쓰는 위치:

| 단계 | 주로 보는 파일/폴더 |
|------|---------------------|
| 데이터 분석 | `data/raw/`, `notebooks/`, `reports/DATA_CARD.md`, `data_manifest.json` |
| 전처리 | `scripts/preprocess.py`, `data/processed/`, `reports/DATA_CARD.md`, `data_manifest.json` |
| 모델링 | `scripts/train.py`, `scripts/train_automl.py`, `src/data/`, `src/models/`, `experiments/runs/`, `model_registry.json` |
| 결과 해석 | `experiments/runs/<run_id>/metrics.json`, `predictions.csv`, `confusion_matrix.json`, `leaderboard.csv`, `automl_summary.json`, `reports/EXPERIMENT_REPORT.md`, `reports/ERROR_ANALYSIS.md` |
| 추가 모델링 / 데이터 추가 | 새 config, 새 `data_version`, 새 run 기록, `model_registry.json` |
| Streamlit 시각화 | `streamlit_app.py`, `src/utils/logger.py`, `logs/inference.jsonl`, `model_registry.json`, `experiments/runs/` |

핵심 기록 파일:

- `data_manifest.json`: 데이터 버전, 출처, 크기, checksum, split 기록
- `model_registry.json`: model_id, data_version, config hash, metric, artifact 경로 기록
- `experiments/runs/<run_id>/confusion_matrix.json`: Streamlit에서 선택한 run의 confusion matrix 시각화에 사용
- `experiments/runs/<run_id>/threshold_metrics.csv`: binary classification threshold별 precision/recall/F1 확인
- `experiments/runs/<run_id>/leaderboard.csv`: AutoGluon 후보 모델 비교
- `experiments/runs/<run_id>/automl_summary.json`: AutoGluon 실행 조건과 선택 모델 요약
- `logs/inference.jsonl`: Streamlit 데모 추론 요청, 상태, latency 기록
- `reports/DATA_CARD.md`: 데이터 설명과 품질 위험
- `reports/EXPERIMENT_REPORT.md`: 주요 실험의 가설, 결과, 해석
- `reports/ERROR_ANALYSIS.md`: 실패 유형과 개선 후보
- `reports/MODEL_CARD.md`: 최종 모델 사용 조건과 한계

---

## 5. 작업 원칙

### 데이터

- 원본 데이터와 가공 데이터는 Git에 올리지 않습니다.
- 데이터가 바뀌면 `data_manifest.json`의 `data_version`, row/column 수, split, checksum을 갱신합니다.
- train/validation/test split 이후에는 test 데이터를 전처리 fit이나 모델 선택에 사용하지 않습니다.
- leakage가 의심되면 모델 성능 개선보다 누수 확인을 먼저 합니다.
- 학생에게 YAML 직접 수정을 요구하지 않습니다. raw 파일명, target, output, data_version은 대화나 slash command 인자로 받습니다.
- EDA는 바로 실행하기보다 먼저 확인할 질문과 표/그래프를 짧게 계획한 뒤 notebook을 만들고 실행합니다.
- `reports/DATA_CARD.md`는 EDA 이후에 데이터 품질, 누수 위험, 전처리 결정을 정리하는 산출물입니다.
- `data/processed/`에는 모델 학습용으로 정돈된 CSV를 저장합니다.
- 컬럼명 정리, rename/drop, target mapping, 중복 제거, target 결측 제거는 `scripts/preprocess.py`에서 처리할 수 있습니다.
- scaler, encoder, imputer처럼 fit이 필요한 변환은 processed 파일에 미리 반영하지 않고 train split 이후 pipeline에서 처리합니다.

### 모델링

- 복잡한 모델보다 단순한 baseline을 먼저 만듭니다.
- 새 실험은 반드시 가설, 변경점, 비교 baseline, 성공/실패 기준을 가집니다.
- config, metric, prediction sample, artifact path가 재현 가능하게 남아야 합니다.
- 실제 프로젝트 baseline은 `python scripts/train.py --data ... --target ... --data-version ...` 형태의 CLI 인자를 우선 사용합니다.
- task type, positive class, primary metric, auxiliary metrics를 먼저 정하고 CLI 인자로 명시합니다.
- Data Card의 split 추천을 맞출 때는 YAML을 수정하지 말고 `--test-size`, `--val-size`, `--no-stratify` CLI 인자를 우선 사용합니다.
- 실행에 사용된 effective config는 각 run의 `experiments/runs/<run_id>/config.yaml`에 자동 저장합니다.
- `predictions.csv`에 feature와 `original_index`가 있으면 오류 분석에 사용합니다. 없으면 feature 값을 추측하지 말고 config와 seed로 split을 재현하거나 한계를 명시합니다.
- 실험별 `confusion_matrix.json`을 남겨 Streamlit에서 모델별 오류 패턴을 확인할 수 있어야 합니다.
- binary classification이고 positive class와 probability가 있으면 `threshold_metrics.csv`를 남깁니다.
- AutoML은 AutoGluon Tabular만 사용하며, baseline 이후 `/plan-automl`에서 metric, split, leakage 제외 컬럼, 성공 기준을 고정한 뒤 실행합니다.
- AutoGluon 실행 전에는 `from autogluon.tabular import TabularPredictor` import가 되는지 확인합니다.
- baseline과 AutoGluon은 같은 processed CSV, target column, `data_version`, split 비율, seed, primary metric을 사용해야 직접 비교할 수 있습니다.
- AutoGluon에는 train과 validation만 전달하고, test set은 최종 `evaluate()`와 leaderboard 확인에만 사용합니다.
- AutoGluon 결과는 전처리 + 모델 + 하이퍼파라미터 pipeline 후보 비교로 해석합니다.
- 모델 파일은 `models/` 또는 외부 저장소에 두고 Git에 올리지 않습니다.

### 결과 해석

- metric 숫자만 보고 결론을 내리지 않습니다.
- data_version이나 split이 다른 실험은 직접 비교하지 않습니다.
- class imbalance, threshold, error type, 모델 한계를 함께 봅니다.
- 최종 Streamlit 화면에서는 Overview, Leaderboard, Evaluation, Prediction, Logs 탭으로 baseline/AutoGluon 결과와 추론 로그를 함께 확인할 수 있어야 합니다.
- global feature importance, global SHAP, batch prediction은 현재 MVP 범위에서 제외합니다. 단, AutoGluon 내부 모델 2개를 같은 입력 row로 비교하는 local SHAP explanation은 Prediction 탭에서 지원합니다.
- 단일 row explanation은 모델 판단의 근거 후보이지 인과 설명이 아니므로 결과 해석에 한계를 함께 표시합니다.

### Claude 사용

- Claude가 만든 코드는 학생이 실행하고 이해한 뒤 반영합니다.
- API key, 개인정보, 민감한 원본 샘플은 프롬프트에 넣지 않습니다.
- 새 라이브러리는 꼭 필요한 경우에만 제안하고 관련 requirements 파일을 함께 갱신합니다. 이미 requirements에 있는 패키지는 `pip install -r requirements-automl.txt`로 설치하도록 안내하고, 특정 패키지를 최신 버전으로 따로 설치하지 않습니다.
- 큰 작업은 계획 → 구현 → 실행 확인 → 문서 갱신 순서로 진행합니다.
- 결과가 좋아 보여도 data leakage, split 변경, metric 선택 문제를 먼저 점검합니다.
- 중복, 누수, 오류 원인을 말할 때는 반올림된 출력이나 row 번호만 근거로 단정하지 않고 실제 파일에서 exact duplicate, feature-only duplicate, split 정보를 확인합니다.

---

## 6. Claude Code 권장 커맨드

| 커맨드 | 언제 사용 | 참조 기준 |
|--------|-----------|-----------|
| `/eda` | raw 파일 기준 EDA 계획과 notebook 실행 | `PROJECT_SPEC.md`, `DATA_ANALYSIS_SPEC.md` |
| `/preprocess-data` | raw 파일을 processed CSV로 정리 | `DATA_CARD.md`, `DATA_ANALYSIS_SPEC.md` |
| `/train-baseline` | processed CSV로 baseline 학습 | `MODELING_SPEC.md`, `EXPERIMENT_REPORT.md` |
| `/plan-automl` | AutoGluon 실행 전 metric/split/leakage/성공 기준 고정 | `MODELING_SPEC.md`, `METRICS_AND_INTERPRETATION_SPEC.md` |
| `/check-streamlit` | Streamlit 실행 전 모델/run/log 연결 점검 | `PROJECT_SPEC.md`, `model_registry.json`, `experiments/runs/`, `streamlit_app.py` |
| `/checkpoint` | 단계 완료 후 문서/코드 변경 checkpoint commit | `CLAUDE.md`, `.gitignore`, `git status` |
| `/plan-experiment` | 새 feature/model 실험 전 | `CLAUDE.md`, `MODELING_SPEC.md`, `METRICS_AND_INTERPRETATION_SPEC.md` |
| `/log-experiment` | 학습 실행 후 | `CLAUDE.md`, `MODELING_SPEC.md` |
| `/compare-models` | 여러 run 비교 | `CLAUDE.md`, `METRICS_AND_INTERPRETATION_SPEC.md` |
| `/analyze-errors` | 오답 원인 분석 | `DATA_ANALYSIS_SPEC.md`, `METRICS_AND_INTERPRETATION_SPEC.md` |
| `/log-inference` | 데모/배포 후 로그 점검 | `CLAUDE.md`, `PROJECT_SPEC.md` |

새 command 파일을 추가한 직후 현재 Claude Code 세션에서 `/command`가 인식되지 않을 수 있습니다.
그 경우 프로젝트 루트에서 새 세션을 열거나 command 파일 내용을 붙여 넣어 동일한 요청으로 처리합니다.
`/checkpoint`는 사용자 승인 전에는 `git add`나 `git commit`을 실행하지 않습니다.

---

## 7. 금지 사항

- 데이터 파일, 모델 가중치, `.env`, Streamlit secrets를 Git에 추가하지 않습니다.
- test set 성능을 반복적으로 보고 모델 선택에 사용하지 않습니다.
- 실패한 실험을 삭제하거나 숨기지 않습니다.
- metric 개선 근거 없이 “성능 향상”이라고 쓰지 않습니다.
- spec 문서와 registry를 갱신하지 않은 채 최종 모델만 남기지 않습니다.
- 새로 발견한 함정이나 프로젝트 규칙은 `CLAUDE.md` 또는 관련 spec에 반영합니다.
