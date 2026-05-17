---
description: "단계 완료 후 변경 파일을 검토하고 git checkpoint를 만듭니다."
argument-hint: "[commit message]"
disable-model-invocation: true
---

# /checkpoint

프로젝트 단계가 끝났을 때 Git에 올릴 변경과 제외할 산출물을 점검합니다.

$ARGUMENTS

## 1. 확인

아래를 먼저 실행해 상태를 확인합니다.

```bash
git status --short
git diff --check
```

필요하면 Python 파일 문법도 확인합니다.

```bash
python -m py_compile scripts/preprocess.py scripts/train.py scripts/train_automl.py streamlit_app.py
```

## 2. 제외할 항목

아래는 Git에 추가하지 않습니다.

- `data/raw/*`
- `data/processed/*`
- `models/*`
- `experiments/runs/*`
- `logs/*`
- `.env`
- `.streamlit/secrets.toml`

## 3. 커밋

사용자 승인 없이 `git add`나 `git commit`을 실행하지 않습니다.
커밋 메시지가 없으면 변경 목적을 보고 짧은 메시지를 제안합니다.

## 4. 기록

커밋 후에는 `reports/PROJECT_REPORT.md`의 Final Checklist 중 완료한 항목을 확인하라고 안내합니다.
