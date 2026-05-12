# Claude Code 활용 가이드

> Claude Code의 목적은 코드를 대신 맡기는 것이 아니라, 데이터와 모델을 더 꼼꼼하게 검토하도록 돕는 것입니다.
> 항상 실행 결과와 근거를 사람이 확인하세요.

---

## 1. 좋은 활용 방향

AI에게 요청하기 전에 먼저 `CLAUDE.md`와 `docs/specs/`의 관련 기준을 확인하세요.
commands는 작업 절차를 돕고, specs는 판단 기준을 제공합니다.

### EDA 관점 넓히기

AI에게 다음을 요청하세요.

```markdown
이 데이터셋의 컬럼 설명, 결측치/이상치 요약, target 분포를 바탕으로
추가로 확인해야 할 EDA 질문 10개를 제안해주세요.
데이터 누수 가능성이 있는 컬럼도 따로 의심 목록으로 정리해주세요.
```

### Feature 후보 만들기

```markdown
현재 문제 정의와 EDA 결과를 바탕으로 feature engineering 후보를 제안해주세요.
각 feature마다 기대 효과, 데이터 누수 위험, 구현 난이도를 표로 정리해주세요.
```

### Metric 해석 검토받기

```markdown
아래 실험 결과에서 metric이 실제 문제 정의와 잘 맞는지 검토해주세요.
accuracy만 볼 때 놓칠 수 있는 위험과 추가로 봐야 할 지표를 제안해주세요.
```

### Error analysis 정리하기

```markdown
오답 샘플과 예측 확률을 보고 실패 유형을 묶어주세요.
각 유형별 원인 가설과 다음 실험 아이디어를 제안해주세요.
```

---

## 2. Command와 자연어 질문 구분

전체 프로젝트 순서는 `README.md`의 `0. 처음 시작하는 순서`를 따릅니다.
이 문서는 그 흐름을 반복하기보다, Claude에게 어떤 방식으로 물어보면 좋은지에 집중합니다.

- 산출물이 정해진 작업은 slash command를 사용합니다: EDA, 전처리, baseline 학습, Streamlit 점검, checkpoint.
- 판단과 해석이 중요한 작업은 자연어로 질문합니다: metric 의미, 오류 원인, feature 후보, 다음 실험 우선순위.
- 패키지가 없을 때는 특정 패키지를 따로 최신 설치하지 말고 `pip install -r requirements.txt`를 우선 사용합니다.

예:

```markdown
방금 결과에서 macro F1과 accuracy가 거의 같은 이유를 설명해줘.
이 데이터에서 다음 실험으로 무엇을 먼저 하는 게 좋은지도 근거와 함께 제안해줘.
```

---

## 3. 상황별 요청 예시

### 처음 데이터 분석 요청

```markdown
/eda data/raw/[파일명] target=[target컬럼명 또는 후보]

PROJECT_SPEC.md와 DATA_ANALYSIS_SPEC.md를 읽고,
먼저 어떤 EDA를 할지 짧게 계획한 뒤 notebook을 만들고 실행해주세요.
결과는 DATA_CARD.md에 옮길 수 있게 정리해주세요.
```

### 전처리 요청

```markdown
/preprocess-data data/raw/[파일명] target=[target컬럼명] output=data/processed/[데이터버전]_modeling.csv

EDA 결과를 바탕으로 모델 학습용 processed 데이터를 만들어주세요.
fit이 필요한 scaler/encoder/imputer는 하지 말고 train 단계에 남겨주세요.
data_manifest.json과 DATA_CARD.md에 남길 요약도 작성해주세요.
```

### Baseline 학습 요청

```markdown
/train-baseline data/processed/[데이터버전]_modeling.csv target=[target컬럼명] data_version=[데이터버전] test_size=0.2 val_size=0.2

학습 전에 EXPERIMENT_REPORT.md에 baseline 계획을 짧게 정리한 뒤 실행해주세요.
DATA_CARD.md의 split 추천과 실행 split이 같도록 CLI 인자를 맞춰주세요.
실행 후 metric, confusion matrix, 한계, 다음 실험 후보를 정리해주세요.
```

### Streamlit 점검 요청

```markdown
/check-streamlit

Streamlit 앱을 실행하기 전에 model_registry, model artifact,
metrics, confusion matrix, streamlit_app.py, logs 상태를 점검해주세요.
여러 모델 record가 있으면 sidebar에서 모델 버전을 선택할 수 있는지도 확인해주세요.
streamlit이 없으면 `pip install -r requirements.txt` 기준으로 설치를 안내해주세요.
문제가 없으면 직접 streamlit run streamlit_app.py를 실행하고 접속 URL을 알려주세요.
기본 포트 8501이 사용 중이면 다른 포트로 실행해주세요.
앱에서 확인할 항목도 체크리스트로 정리해주세요.
```

### Checkpoint commit 요청

```markdown
/checkpoint entropy 제거 실험 결과 기록

git status와 diff stat을 먼저 보여주고,
문서/코드/가벼운 JSON만 stage 후보로 정리해주세요.
data/raw, data/processed, models, logs, experiments/runs는 기본 제외해주세요.
stage/commit 전에 반드시 확인 질문을 해주세요.
```

---

## 4. 요청 템플릿

```markdown
## Goal
[이번 분석/실험의 목표를 한 문장으로]

## Current Evidence
- 데이터 버전: [data_manifest.json의 data_version]
- 현재 baseline: [model_registry.json의 model_id / 주요 metric]
- 관찰한 문제: [예: 특정 class recall이 낮음]
- 참고 spec: [docs/specs/...]

## Constraints
- 데이터 누수 가능성을 우선 확인해주세요.
- 새 라이브러리는 꼭 필요한 경우에만 제안해주세요.
- metric 개선뿐 아니라 해석 가능성과 재현성도 함께 봐주세요.

## Task
[EDA 관점 제안 / feature 후보 / error analysis / metric 해석 등]

## Expected Output
- 확인해야 할 질문
- 실험 가설
- 구현 전 주의점
- 성공/실패 판단 기준
```

---

## 5. 반드시 사람이 확인할 것

- train/validation/test split 이후 전처리 fit이 train에만 적용됐는가?
- target 또는 미래 정보를 담은 컬럼이 feature에 섞이지 않았는가?
- metric이 문제의 비용 구조와 맞는가?
- seed, config, data version이 실험 로그에 남았는가?
- AI가 만든 해석이 실제 데이터 분포와 맞는가?
- 중복, 누수, 오류 샘플 원인은 실제 파일과 exact 계산으로 확인했는가?

---

## 6. README에 남길 AI 사용 내역

```markdown
## AI Tool Usage

| 항목 | 내용 |
|------|------|
| 사용한 도구 | Claude Code |
| 사용 목적 | EDA 질문 생성, feature 후보 검토, 오류 분석 보조 |
| AI가 제안한 내용 | [예: 누수 의심 컬럼 목록, threshold 조정 실험] |
| 사람이 검토하고 수정한 내용 | [예: 실제 데이터 확인 후 제외한 제안, 채택한 실험] |
```
