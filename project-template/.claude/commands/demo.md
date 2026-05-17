---
description: "Streamlit 데모, prediction, inference log를 점검합니다."
argument-hint: "[run_id or model_id]"
disable-model-invocation: true
---

# /demo

Streamlit 앱에서 모델/run/log가 연결되는지 확인하고 데모 준비 상태를 점검합니다.

$ARGUMENTS

## 1. 확인할 기준

- `PROJECT_GUIDE.md`의 Streamlit Demo 기준
- `reports/PROJECT_REPORT.md`의 Demo & Monitoring 섹션
- `model_registry.json`
- `experiments/runs/`
- `logs/inference.jsonl`
- `streamlit_app.py`

## 2. 실행 전 점검

- 선택할 model_id와 artifact_path가 존재하는가?
- AutoGluon이면 `leaderboard.csv`, `automl_summary.json`이 있는가?
- classification이면 `confusion_matrix.json`과 threshold metrics를 확인할 수 있는가?
- prediction 탭의 입력 컬럼이 모델 feature와 맞는가?

## 3. 실행

직접 실행이 필요하면 아래 명령을 제안합니다.

```bash
streamlit run streamlit_app.py
```

포트 충돌 시 `--server.port 8502`를 사용합니다.

## 4. 기록

Demo & Monitoring 섹션에 아래를 남깁니다.

- 확인한 탭: Overview, Leaderboard, Evaluation, Prediction, Logs
- inference log 생성 여부
- latency/error rate 확인 결과
- 데모에서 보여줄 최종 모델
- 알려야 할 한계
