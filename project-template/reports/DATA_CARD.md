# Data Card

## 1. 데이터 개요

- **데이터 이름**:
- **데이터 버전**:
- **출처 / 링크**:
- **수집 기간 / 조건**:
- **라이선스 / 사용 조건**:
- **raw 파일 경로**:
- **processed 파일 경로**:
- **실행한 EDA notebook**:

## 2. 구조

- **row 수**:
- **column 수**:
- **target column**:
- **주요 feature**:

| 컬럼 | 의미 | 타입 | 결측치 | 비고 |
|------|------|------|--------|------|
|      |      |      |        |      |

## 3. EDA 핵심 발견

- 확인한 주요 질문:
- target 분포:
- feature 분포:
- feature redundancy:
- 결측치 / 이상치:
- 중복 데이터:
- 눈에 띄는 패턴:
- split 방식 추천:

## 4. Feature Drop Candidates

| column | reason | evidence | recommended_action |
|--------|--------|----------|--------------------|
|        |        |          |                    |

- **correlation threshold**: numeric feature 간 절대 상관계수 `>= 0.95`
- **high-missing threshold**: 결측률 `>= 80%`
- **near-constant threshold**: 최빈값 비율 `>= 95%`
- **preprocess 반영 여부**:

## 5. 데이터 품질과 위험

- 라벨 오류 가능성:
- 편향 가능성:
- 개인정보 / 민감정보:
- 데이터 누수 의심 컬럼:

## 6. 전처리 결정

- 적용한 전처리:
- 제거한 컬럼:
- 제거한 이유와 근거:
- rename / target mapping:
- processed에 반영하지 않고 train pipeline에 남긴 변환:
- 전처리 후 row / column 수:

### Pipeline Decisions

아래 YAML block은 `python scripts/preprocess.py --decisions reports/DATA_CARD.md` 실행 시 자동 적용됩니다.
자유 서술이나 표는 기록용이고, 실제 실행 설정은 이 block만 사용합니다.

```yaml pipeline_decisions
input: ""
output: ""
target: ""
data_version: ""
sep: ","
header: infer
columns: []
drop_columns: []
rename: {}
target_map: {}
drop_duplicate_rows: false
keep_rows_missing_target: false
manifest: data_manifest.json
source: ""
notes: ""
```

## 7. Split과 버전 기록

- data_version:
- train / validation / test 비율:
- split seed:
- raw checksum:
- processed checksum:
- `data_manifest.json` 반영 여부:

## 8. 다음 액션

- [ ] 추가 확인할 EDA 질문
- [ ] 정제하거나 제외할 컬럼
- [ ] 추가 수집 또는 라벨 검토 필요 사항
