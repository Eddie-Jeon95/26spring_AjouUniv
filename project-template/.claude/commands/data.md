---
description: "EDA, drop 결정, preprocessing decision, processed CSV 생성을 한 단계로 진행합니다."
argument-hint: "[raw_path] target=[target]"
disable-model-invocation: true
---

# /data

raw 데이터 기준 EDA를 수행하고, 학생이 drop/전처리 결정을 확인한 뒤 `pipeline_decisions`로 preprocessing을 실행합니다.

$ARGUMENTS

## 1. 확인할 기준

- `PROJECT_GUIDE.md`의 Data Analysis, Leakage, Processed Data 기준
- `reports/PROJECT_REPORT.md`의 Data Card와 `pipeline_decisions`
- 인자로 받은 raw 파일, target, separator/header 후보

raw 파일이 없으면 중단합니다. 예제 데이터를 자동 생성하거나 다운로드하지 않습니다.

## 2. EDA 계획

notebook을 만들기 전에 아래 계획을 5~10줄로 제시합니다.

```markdown
## EDA Plan
- 데이터 로딩 방식:
- target 확인:
- 기본 구조:
- target 분포:
- 결측치 / 중복 / 이상치:
- feature redundancy:
- 누수 의심:
- split 방식 후보:
```

## 3. 학생 Decision Gate

EDA 결과 후 바로 preprocessing하지 말고 아래 표를 제시하고 학생 결정을 받습니다.

| 항목 | AI 추천 | 근거 | 학생 결정 |
|------|---------|------|-----------|
| drop columns | | | keep/drop/investigate |
| duplicate row | | | keep/drop |
| target missing row | | | drop/keep |
| split 방식 | | | random/stratified/time/group |
| 추가 확인 컬럼 | | | |

## 4. 실행

결정이 끝나면 `reports/PROJECT_REPORT.md`의 `pipeline_decisions` block을 채우고 아래 명령을 실행합니다.

```bash
python scripts/preprocess.py --decisions reports/PROJECT_REPORT.md
```

CLI 인자는 임시 override가 필요할 때만 사용합니다.

## 5. 기록

`reports/PROJECT_REPORT.md`의 Data Card 섹션에 아래를 정리합니다.

- row / column 수
- target 분포
- 결측치, 중복, 이상치
- drop 후보와 최종 결정
- leakage 의심 컬럼
- processed 파일 경로와 data_version
- `data_manifest.json` 반영 여부
