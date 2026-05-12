---
description: "Streamlit 앱과 모델 registry, run artifact, 추론 로그 연결 상태를 점검합니다."
argument-hint: "[port=8501 선택사항]"
disable-model-invocation: true
---

# /check-streamlit

Streamlit 앱 실행 전후 연결 상태를 점검하는 커맨드입니다.
앱을 새로 구현하거나 모델을 다시 학습하지 않고, 이미 생성된 registry, run artifact, 모델 파일, 로그 파일이 Streamlit 앱과 잘 이어지는지 확인합니다.

---

입력 예시:

```text
/check-streamlit
```

$ARGUMENTS

## 1단계: 기준 문서와 앱 목표 확인

- `CLAUDE.md`의 Streamlit 시각화 기준 확인
- `docs/specs/PROJECT_SPEC.md`의 최종 Streamlit 대시보드 기준 확인
- `streamlit_app.py`가 존재하는지 확인
- 앱이 최소한 다음 탭/기능을 제공하는지 확인
  - Predict: feature 입력, 예측, 추론 로그 기록
  - Experiments: 모델 버전별 metric, 선택 run의 confusion matrix
  - Logs: 추론 요청 수, 에러율, 평균/P95 latency, 최근 로그

## 2단계: 모델 버전 registry 확인

- `model_registry.json`을 읽고 record 수를 확인
- 각 record의 핵심 필드 확인
  - `model_id`
  - `run_id`
  - `data_version`
  - `primary_metric`
  - `metrics`
  - `artifact_path`
  - `experiment_path`
  - `confusion_matrix_path`
- record가 2개 이상이면 sidebar에서 여러 모델/run을 선택할 수 있는 구조인지 확인
- 같은 `data_version`끼리 metric 추이를 비교할 수 있는지 확인
- `data_version`이 다른 모델은 직접 우열 비교하면 안 된다고 안내

## 3단계: 선택 가능한 각 모델 산출물 확인

최신 record를 기본으로 보되, 가능하면 모든 record에 대해 확인합니다.

- `artifact_path`의 `.pkl` 모델 파일 존재 여부
- `experiment_path/metrics.json` 존재 여부
- `confusion_matrix_path` 존재 여부
- metric 값이 `model_registry.json`과 `metrics.json`에서 일관적인지 확인
- 모델 파일이 없으면 재학습 또는 artifact 경로 확인을 안내

## 4단계: 정적 검증

다음 명령으로 앱 문법과 import를 확인합니다.

```bash
python -m py_compile streamlit_app.py
python -m py_compile src/utils/logger.py
python -c "import streamlit; print(streamlit.__version__)"
python -c "from src.utils.logger import InferenceLogger; print('logger OK')"
```

`streamlit`이 설치되어 있지 않으면 아래를 안내합니다.

```bash
pip install -r requirements.txt
```

주의:
- `pip install streamlit`처럼 패키지 하나만 최신 버전으로 설치하지 않습니다. 템플릿은 `requirements.txt`의 고정 버전을 기준으로 재현성을 맞춥니다.
- 이미 다른 Streamlit 버전이 설치되어 실행된다면 버전 차이를 짧게 알리고, 필요할 때만 `pip install -r requirements.txt`를 권합니다.

## 5단계: 앱 실행

정적 검증과 산출물 확인이 끝나면 가능하면 직접 앱을 실행합니다.

```bash
streamlit run streamlit_app.py
```

기본 포트는 `8501`을 사용합니다.
이미 사용 중이면 사용 가능한 다른 포트를 찾아 아래처럼 실행합니다.

```bash
streamlit run streamlit_app.py --server.port 8502
```

실행에 성공하면 사용자에게 접속 URL을 알려주세요.

예:

- Local URL: `http://localhost:8501`
- 포트를 바꿨다면: `http://localhost:<port>`

실행 세션이 계속 떠 있어야 하므로, 실행 후에는 터미널 세션을 종료하지 말고 URL과 확인 항목을 안내합니다.

앱에서 확인할 항목:

- sidebar에서 모델/run 선택 가능 여부
- Predict 탭에서 feature 4개 입력 후 예측 가능 여부
- 예측 결과에 class label과 위조 확률이 표시되는지
- Experiments 탭에서 metric 표와 confusion matrix가 보이는지
- Logs 탭에서 로그가 없을 때 빈 상태 안내가 보이는지
- Predict를 1회 실행한 뒤 `logs/inference.jsonl`이 생성되는지

브라우저 조작이 어려운 환경이면 앱과 같은 모델 artifact를 Python에서 직접 로드해 기본 입력값으로 smoke prediction을 1회 수행하고, `InferenceLogger`로 동일 schema의 success log가 남는지 확인합니다.
이 경우 “UI 클릭 검증은 미수행, 로직/로그 연결만 검증”이라고 명확히 말합니다.

## 6단계: 로그 점검 연결

- `logs/inference.jsonl`이 없으면 Predict 탭에서 예측을 1회 실행하라고 안내
- 로그가 있으면 `/log-inference`로 운영 로그 요약 분석을 이어가라고 안내

---

주의:
- 이 커맨드는 앱 점검용입니다. 새 모델 학습은 `/train-baseline` 또는 `/plan-experiment` 이후 진행하세요.
- 로그 원문에 개인정보나 민감 정보가 있으면 그대로 출력하지 말고 존재 여부만 알리세요.
- Streamlit Cloud 배포 설정은 이 커맨드의 기본 범위가 아닙니다.
- 로컬 앱 실행이 권한, 패키지 설치, 포트 문제로 실패하면 실패 원인과 다음 명령을 짧게 알려주세요.
