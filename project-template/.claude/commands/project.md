---
description: "문제 정의와 성공 기준을 정리하고 PROJECT_REPORT를 시작합니다."
argument-hint: "[project goal or notes]"
disable-model-invocation: true
---

# /project

프로젝트 시작 단계에서 문제 정의, target, task type, metric, 오류 비용, 데모 목표를 고정합니다.

$ARGUMENTS

## 1. 확인할 기준

- `CLAUDE.md`의 작업 원칙
- `PROJECT_GUIDE.md`의 문제 정의, metric, 데모 기준
- `reports/PROJECT_REPORT.md`의 Project Definition 섹션

## 2. 학생에게 확인할 핵심 결정

아래 항목은 추측하지 말고 후보를 제시한 뒤 학생이 고르게 합니다.

- 예측 대상과 target column
- task type: classification / regression
- binary classification의 positive class
- 실제 사용자가 더 피해야 하는 오류: false positive / false negative / 큰 오차
- primary metric 1개와 auxiliary metric 1~3개
- Streamlit 최종 화면에서 꼭 보여줄 항목

## 3. 기록 형식

`reports/PROJECT_REPORT.md`의 Project Definition 섹션에 아래를 채울 수 있게 정리합니다.

```markdown
## Project Definition
- 문제 정의:
- 입력 데이터:
- target column:
- task type / positive class:
- 사용자와 사용 상황:
- 중요한 오류:
- primary metric / auxiliary metrics:
- 성공 기준:
- Streamlit 확인 항목:
```

## 4. 다음 단계

문제 정의가 충분하면 다음 액션으로 `/data data/raw/[raw_file] target=[target]`를 제안합니다.
데이터가 없으면 `data/raw/`에 원본 파일을 넣고 다시 시작하도록 안내합니다.
