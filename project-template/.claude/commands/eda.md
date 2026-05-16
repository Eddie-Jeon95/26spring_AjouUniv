---
description: "raw 데이터 기준 EDA 계획을 세우고 notebook 실행과 Data Card 요약을 돕습니다."
argument-hint: "[raw_path] target=[target]"
disable-model-invocation: true
---

# /eda

raw 데이터 파일을 기준으로 EDA 계획을 짧게 세우고, notebook을 만들어 실행한 뒤 `reports/DATA_CARD.md`에 옮길 요약을 정리하는 커맨드입니다.

---

입력 인자 예시:

```text
/eda data/raw/[raw_file] target=[target]
```

$ARGUMENTS

## 1단계: 기준 문서와 파일 확인

- `CLAUDE.md`의 데이터 작업 원칙 확인
- `docs/specs/PROJECT_SPEC.md`의 문제 정의와 target 후보 확인
- `docs/specs/DATA_ANALYSIS_SPEC.md`의 EDA 기준 확인
- 인자로 받은 raw 파일 경로가 존재하는지 확인
- raw 파일이 없으면 여기서 즉시 중단하고 notebook을 만들지 않습니다.
  - 예제 데이터(seaborn/sklearn/Kaggle sample 등)를 자동 생성하거나 다운로드하지 않습니다.
  - 대신 `data/raw/`에 파일을 넣기, 경로 재확인, `target=...` 재지정을 다음 액션으로 안내합니다.
- `target=...`이 있으면 우선 사용하고, 없거나 불확실하면 target 후보를 질문
- header가 없는 파일이면 컬럼명 후보를 질문하거나 데이터 첫 줄을 보고 임시 컬럼명을 제안

## 2단계: EDA 계획을 먼저 제시

바로 notebook을 만들기 전에 다음을 5~10줄로 짧게 계획하세요.

```markdown
## EDA Plan
- 데이터 로딩 방식:
- target 확인:
- 기본 구조:
- target 분포:
- 결측치 / 중복 / 이상치:
- feature 타입:
- feature redundancy / drop candidates:
- 누수 의심:
- split 방식 후보:
```

## 3단계: notebook 생성과 실행

- `notebooks/01_eda_<dataset>.ipynb`를 생성하거나 갱신
- raw 파일 경로와 target column을 notebook 상단에 명시
- 최소 확인 항목:
  - shape, info, describe
  - target value counts / plot
  - missing ratio
  - duplicate rows
  - numeric/category/text/date feature 타입 요약
  - 주요 numeric 분포
  - numeric correlation heatmap
  - high-correlation pair table (`abs(correlation) >= 0.95`)
  - ID-like/high-cardinality column check
  - high-missing column check (`missing_rate >= 0.80`)
  - constant / near-constant column check
  - target과 feature 관계의 초기 단서
  - 누수 의심 컬럼 메모
- 가능한 경우 notebook을 실행해 output을 남김

## 4단계: Data Card 반영 요약

`reports/DATA_CARD.md`에 옮길 수 있게 다음 형식으로 요약하세요.

```markdown
## DATA_CARD 업데이트 초안
- raw 파일 경로:
- 실행한 EDA notebook:
- row / column 수:
- target column:
- target 분포:
- 결측치 / 이상치 / 중복:
- 주요 feature:
- drop 후보:
- drop 이유:
- 근거:
- preprocess 반영 여부:
- 누수 의심 컬럼:
- split 방식 추천:
- 전처리 후보:
```

---

주의:
- 개인정보나 민감한 원본 샘플을 그대로 출력하지 마세요.
- EDA는 그래프 수보다 “그래프에서 무엇을 판단했는가”를 우선합니다.
- raw 파일이 없을 때는 대체 데이터로 진행하지 마세요. 데이터 출처와 기록이 오염됩니다.
- `DATA_CARD.md`를 직접 수정했다면 어떤 섹션을 바꿨는지 마지막에 알려주세요.
