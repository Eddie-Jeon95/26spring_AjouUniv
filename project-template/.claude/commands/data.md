---
description: "EDA, drop/feature engineering 결정, preprocessing decision, processed CSV 생성을 한 단계로 진행합니다."
argument-hint: "[raw_path] target=[target]"
disable-model-invocation: true
---

# /data

raw 데이터 기준 EDA를 수행하고, 학생이 drop/feature engineering/전처리 결정을 확인한 뒤 `pipeline_decisions`로 preprocessing을 실행합니다.

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

## 4. Feature Engineering Decision Gate

drop 결정과 별도로, 모델에 도움이 될 수 있는 파생 feature 후보를 제안합니다.
단, target이나 validation/test 성능을 보고 고른 feature는 제안하지 않습니다.

| feature 후보 | source columns | 기대 효과 | leakage/예측시점 위험 | AI 추천 | 학생 결정 |
|--------------|----------------|-----------|------------------------|---------|-----------|
|              |                |           |                        |         | create/skip/investigate |

허용 예시:

- 날짜에서 `year`, `month`, `weekday` 파생
- 텍스트 길이, 단어 수, 비어 있음 여부
- 숫자 비율, 차이, 변화량, rolling 없이 한 row 안에서 계산 가능한 요약
- 단위 변환, 명확한 boolean flag, category grouping

금지 또는 보류:

- target encoding
- scaler, encoder, imputer처럼 train split에 fit해야 하는 변환
- train/test 전체를 보고 고른 supervised feature selection
- validation/test 성능을 본 뒤 선택한 feature
- 예측 시점에 알 수 없는 사후 정보

학생이 `create`로 승인한 feature만 `scripts/preprocess.py`의 `add_project_features(df)` 함수에 읽기 쉬운 Python 코드로 추가합니다.
범용 YAML recipe나 숨은 자동 feature 생성 규칙은 만들지 않습니다.

## 5. 실행 전 추가 확인

아래 상황에서는 preprocess를 계속하지 말고 학생에게 확인을 받습니다.

- feature 생성에 필요한 source column이 없음
- 승인된 feature가 leakage 가능성이 높아 보임
- feature 생성 후 결측치가 대량 발생할 것으로 예상됨
- row 수가 예상보다 크게 줄어들 수 있음
- target column 또는 핵심 컬럼 타입이 예상과 다름
- feature engineering으로 새 processed 데이터가 되는데 `data_version`이 아직 정해지지 않음

단순한 컬럼명 공백, 명확한 타입 변환 가능 여부처럼 수정안이 분명한 문제는 수정안을 짧게 제시하고 확인 후 진행합니다.

## 6. 실행

결정이 끝나면 `reports/PROJECT_REPORT.md`의 `pipeline_decisions` block을 채우고 아래 명령을 실행합니다.

```bash
python scripts/preprocess.py --decisions reports/PROJECT_REPORT.md
```

CLI 인자는 임시 override가 필요할 때만 사용합니다.

## 7. 기록

`reports/PROJECT_REPORT.md`의 Data Card 섹션에 아래를 정리합니다.

- row / column 수
- target 분포
- 결측치, 중복, 이상치
- drop 후보와 최종 결정
- feature engineering 후보, 학생 결정, 실제 구현 여부
- leakage 의심 컬럼
- processed 파일 경로와 data_version
- `data_manifest.json` 반영 여부
