# FAQ

## 데이터 파일이 아직 없는데 학습 스크립트를 실행해도 되나요?

설치 확인 목적이라면 기본 demo fallback으로 `train.py` 실행 흐름을 볼 수 있습니다.
실제 프로젝트에서는 raw 파일을 `data/raw/`에 두고 `/data` 단계에서 processed CSV를 만든 뒤 학습하세요.

## 긴 Python 명령어를 직접 써야 하나요?

기본적으로는 아닙니다.
`reports/PROJECT_REPORT.md`의 decision block을 채운 뒤 아래처럼 실행합니다.

```bash
python scripts/preprocess.py --decisions reports/PROJECT_REPORT.md
python scripts/train.py --decisions reports/PROJECT_REPORT.md
python scripts/train_automl.py --decisions reports/PROJECT_REPORT.md
```

CLI 인자는 임시 override가 필요할 때만 사용합니다.

## decision block은 어디에 있나요?

`reports/PROJECT_REPORT.md` 안에 있습니다.

- `pipeline_decisions`: raw 파일, processed 파일, target, drop columns, rename, target mapping
- `training_decisions`: processed 데이터, task type, metric, split, baseline model, AutoML 설정

자유 서술과 표는 기록용이고, 실제 실행 설정은 이 YAML block만 사용합니다.

## 어떤 slash command를 써야 하나요?

일반적인 순서는 아래와 같습니다.

```text
/project
/data
/train
/automl
/evaluate
/demo
/checkpoint
```

특정 컬럼 판단, metric 해석, 오류 메시지 설명은 command보다 자연어 채팅으로 물어보는 것이 좋습니다.

## 실험 결과는 어디에 기록되나요?

학습 스크립트는 아래를 생성합니다.

- `experiments/runs/<run_id>/config.yaml`
- `experiments/runs/<run_id>/metrics.json`
- `experiments/runs/<run_id>/test_metrics.json`
- `experiments/runs/<run_id>/predictions.csv`
- `experiments/runs/<run_id>/confusion_matrix.json`
- `model_registry.json`

학생이 해석한 내용은 `reports/PROJECT_REPORT.md`에 정리합니다.

## DVC나 MLflow를 꼭 써야 하나요?

아니요.
기본 템플릿은 `data_manifest.json`, `model_registry.json`, `experiments/runs/`만으로 재현에 필요한 정보를 남기도록 설계했습니다.
데이터가 크거나 실험 수가 많아지면 DVC나 MLflow를 선택적으로 검토하세요.

## 모델 파일은 어디에 두나요?

기본 학습 스크립트는 `models/`에 artifact를 저장합니다.
이 폴더는 Git 제외 대상입니다.
데모나 재현에 필요한 모델은 Google Drive, Hugging Face Hub 등 외부 링크를 `model_registry.json`이나 `PROJECT_REPORT.md`에 기록하세요.

## AutoGluon 설치가 오래 걸립니다.

정상입니다.
AutoGluon은 여러 tabular 모델 후보와 optional dependency를 설치하므로 시간이 걸릴 수 있습니다.
반드시 Python 3.10.x 가상환경에서 `pip install -r requirements.txt`로 설치하세요.

## Streamlit 앱은 어떻게 실행하나요?

```bash
streamlit run streamlit_app.py
```

포트가 이미 사용 중이면:

```bash
streamlit run streamlit_app.py --server.port 8502
```

## Git에 올리면 안 되는 것은 무엇인가요?

아래는 Git에 추가하지 않습니다.

- `data/raw/*`
- `data/processed/*`
- `models/*`
- `experiments/runs/*`
- `logs/*`
- `.env`
- `.streamlit/secrets.toml`

## 커밋은 언제 하나요?

큰 단계가 끝날 때 `/checkpoint`를 사용합니다.
예를 들어 project definition, data preprocessing, baseline, AutoML, final evaluation이 각각 좋은 checkpoint입니다.
