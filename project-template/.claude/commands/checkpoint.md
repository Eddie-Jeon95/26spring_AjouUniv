# /checkpoint

작업 단계가 끝났을 때 문서/코드 변경을 검토하고 checkpoint commit을 만드는 커맨드입니다.
MD가 계속 바뀌어 추적이 어려워지는 문제를 줄이되, 검토 전 내용이 자동으로 Git 히스토리에 들어가지 않도록 안전하게 진행합니다.

---

입력 예시:

```text
/checkpoint EDA와 Data Card 초안 기록
/checkpoint entropy 제거 실험 결과 기록
```

$ARGUMENTS

## 1단계: 변경 상태 확인

먼저 아래를 확인하고 사용자에게 요약합니다.

```bash
git status --short
git diff --stat
```

확인할 것:

- 수정된 문서/코드 파일
- 새로 생긴 파일
- Git ignore 대상이어서 커밋되지 않는 데이터/모델/로그 파일
- 현재 변경 중 데이터/모델 산출물이 문서와 맞는지

## 2단계: 기본 stage 후보 분류

기본적으로 stage해도 되는 파일:

- `*.md`
- `docs/specs/*.md`
- `reports/*.md`
- `.claude/commands/*.md`
- `scripts/*.py`
- `src/**/*.py`
- `streamlit_app.py`
- `configs/*.yaml`
- `requirements.txt`
- `data_manifest.json`
- `model_registry.json`

기본적으로 stage하지 않는 파일:

- `data/raw/*`
- `data/processed/*`
- `models/*`
- `logs/*`
- `experiments/runs/*`
- `.env`, `.streamlit/secrets.toml`
- 대용량 이미지/그래프 파일
- notebook output이 많은 `notebooks/*.ipynb`

예외:

- 사용자가 명시적으로 notebook이나 그래프 파일까지 commit하라고 하면, 파일 크기와 민감정보 여부를 먼저 확인한 뒤 포함합니다.
- `.gitkeep`처럼 폴더 구조를 유지하기 위한 파일은 필요하면 포함할 수 있습니다.

## 3단계: stage 전 확인

바로 commit하지 말고 먼저 아래 형식으로 사용자에게 보여줍니다.

```markdown
## Checkpoint 후보

Stage 예정:
- ...

Stage 제외:
- ... (이유)

Commit message:
`[제안 메시지]`

이대로 stage/commit 진행해도 될까요?
```

커밋 메시지 규칙:

- 사용자가 `/checkpoint [메시지]`로 입력한 내용을 우선 사용
- 메시지가 없으면 단계 기준으로 짧게 작성
  - `Record EDA checkpoint`
  - `Record preprocessing checkpoint`
  - `Record baseline experiment checkpoint`
  - `Record Streamlit checkpoint`

## 4단계: stage와 검증

사용자가 승인하면 명시한 파일만 stage합니다.

```bash
git add [승인된 파일 목록]
git diff --cached --stat
git status --short
```

검증할 것:

- 데이터/모델/로그 파일이 stage되지 않았는가?
- 민감정보, API key, `.env`, Streamlit secrets가 stage되지 않았는가?
- commit message가 작업 내용을 설명하는가?

민감정보나 대용량 파일이 stage되었으면 commit하지 말고 이유를 설명한 뒤 stage 목록을 다시 정리합니다.

## 5단계: commit

검증이 끝나면 commit합니다.

```bash
git commit -m "[commit message]"
```

완료 후 다음을 요약합니다.

- commit hash
- commit message
- 포함된 주요 파일
- 제외한 주요 파일과 이유

---

주의:

- 이 커맨드는 “MD 수정마다 즉시 자동 commit”이 아니라 “단계 완료 후 검토 가능한 checkpoint commit”을 만드는 용도입니다.
- 실패한 실험도 기록할 수 있지만, 실험 결과 해석이 문서에 반영됐는지 먼저 확인합니다.
- Git ignore 대상 파일을 강제로 추가하지 않습니다.
- 사용자가 승인하기 전에는 `git add`와 `git commit`을 실행하지 않습니다.
