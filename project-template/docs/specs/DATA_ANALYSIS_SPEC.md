# Data Analysis Spec

데이터 분석의 목표는 예쁜 그래프를 많이 만드는 것이 아니라, 모델링 전에 데이터가 어떤 문제와 위험을 갖는지 판단하는 것입니다.

---

## 1. EDA 목표

EDA는 최소한 다음을 답해야 합니다.

- target은 어떻게 분포되어 있는가?
- 결측치, 이상치, 중복은 모델에 영향을 줄 정도인가?
- feature와 target 사이에 설명 가능한 관계가 있는가?
- 데이터 누수 가능성이 있는 컬럼이 있는가?
- train/validation/test split 방식이 문제 상황과 맞는가?
- 라벨 오류, 편향, 개인정보 위험이 있는가?

---

## 2. 기본 점검 항목

| 항목 | 확인 방법 | 기록 위치 |
|------|-----------|-----------|
| 데이터 크기 | row/column 수 | `DATA_CARD.md`, `data_manifest.json` |
| target 분포 | value counts, histogram | `DATA_CARD.md` |
| 결측치 | column별 missing ratio | `DATA_CARD.md` |
| 중복 | duplicated row, ID 중복 | `DATA_CARD.md` |
| 이상치 | 분포, boxplot, domain rule | `DATA_CARD.md` |
| feature 타입 | numeric/category/text/date | `DATA_CARD.md` |
| checksum | sha256 또는 manifest 기록 | `data_manifest.json` |

---

## 3. Split 기준

무작위 split을 기본으로 하되, 아래에 해당하면 다른 split을 고려합니다.

- 시간 순서가 중요한 문제: 시간 기반 split
- 같은 사용자/상품/문서가 반복 등장: group split
- class imbalance가 큼: stratified split
- 데이터가 매우 적음: cross validation 검토

split을 바꾸면 이전 실험과 metric을 직접 비교하기 어렵습니다. split 변경은 새로운 `data_version`으로 기록하세요.

---

## 4. Data Leakage 점검

누수 의심 예시:

- target을 계산한 뒤 생성되는 컬럼
- 예측 시점에는 알 수 없는 미래 정보
- train/test 전체로 fit한 scaler, encoder, imputer
- 같은 샘플 또는 같은 사용자 데이터가 train/test에 동시에 존재
- 파일명, ID, timestamp가 사실상 label을 암시하는 경우

누수가 의심되면:

1. 의심 컬럼을 `DATA_CARD.md`에 기록합니다.
2. 해당 컬럼을 제거한 baseline과 포함한 baseline을 비교합니다.
3. 성능 차이가 과도하면 누수 가능성을 발표에서 명시합니다.

---

## 5. Data Card 작성 기준

`reports/DATA_CARD.md`에는 최소한 다음을 채웁니다.

- 데이터 출처와 사용 조건
- row/column 수와 target column
- 주요 feature 설명
- target 분포
- 결측치/이상치/중복 요약
- split 방법과 seed
- 누수 의심 컬럼
- 편향, 개인정보, 라벨 품질 위험

그래프를 넣는 것보다 “그래프에서 무엇을 판단했는가”가 중요합니다.

---

## 6. 모델링 전 완료 조건

- [ ] target 분포를 확인했다.
- [ ] split 기준과 seed를 정했다.
- [ ] 누수 의심 컬럼을 점검했다.
- [ ] baseline에 사용할 feature 후보를 정했다.
- [ ] `data_manifest.json`에 데이터 버전을 기록했다.
- [ ] `reports/DATA_CARD.md` 초안을 작성했다.
