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

EDA는 `DATA_CARD.md`를 채우기 전에 먼저 수행합니다.
Claude Code에게 raw 파일명을 알려주면, 먼저 확인할 질문과 표/그래프를 짧게 계획한 뒤 notebook을 만들고 실행합니다.

---

## 2. EDA 실행 기준

EDA notebook에는 최소한 다음이 포함되어야 합니다.

- raw 파일 경로, target column, 데이터 로딩 방식
- row/column 수와 feature 타입 요약
- target 분포
- 결측치, 중복, 이상치 요약
- numeric/category/text/date feature의 기본 분포
- target과 feature 관계의 초기 단서
- 데이터 누수 의심 컬럼
- split 방식 추천과 이유
- `DATA_CARD.md`에 옮길 핵심 판단

Notebook 이름은 raw 데이터 이름을 반영해 `notebooks/01_eda_<dataset>.ipynb`처럼 저장합니다.
기존 `01_eda_template.ipynb`는 참고용 템플릿입니다.

---

## 3. 기본 점검 항목

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

## 4. Split 기준

무작위 split을 기본으로 하되, 아래에 해당하면 다른 split을 고려합니다.

- 시간 순서가 중요한 문제: 시간 기반 split
- 같은 사용자/상품/문서가 반복 등장: group split
- class imbalance가 큼: stratified split
- 데이터가 매우 적음: cross validation 검토

split을 바꾸면 이전 실험과 metric을 직접 비교하기 어렵습니다. split 변경은 새로운 `data_version`으로 기록하세요.
baseline과 AutoGluon은 같은 `data_version`, split 비율, seed, target column을 사용해야 합니다.
test set은 최종 확인용이며, 모델 선택이나 threshold 선택에 사용하지 않습니다.

---

## 5. Data Leakage 점검

누수 의심 예시:

- target을 계산한 뒤 생성되는 컬럼
- 예측 시점에는 알 수 없는 미래 정보
- train/test 전체로 fit한 scaler, encoder, imputer
- 같은 샘플 또는 같은 사용자 데이터가 train/test에 동시에 존재
- 파일명, ID, timestamp가 사실상 label을 암시하는 경우

누수가 의심되면:

1. 의심 컬럼을 `DATA_CARD.md`에 기록합니다.
2. 해당 컬럼을 제거한 baseline과 포함한 baseline을 비교합니다.
3. 성능 차이가 과도하면 누수 가능성을 결과 해석에 명시합니다.

AutoML 전에 추가로 확인합니다.

- AutoGluon에 전달할 processed CSV에서 leakage 제외 컬럼이 실제로 빠졌는가?
- target 이후 생성되는 날짜, 상태값, 결과 요약 컬럼이 남아 있지 않은가?
- train/validation/test split이 baseline과 같은 기준으로 재현되는가?
- test set을 AutoGluon 탐색, 모델 선택, threshold 선택에 사용하지 않는가?
- AutoGluon이 만든 전처리/모델/하이퍼파라미터 조합은 데이터 변경이 아니라 pipeline 변경으로 기록되는가?

---

## 6. Processed 데이터 기준

`data/processed/`에는 모델 학습용으로 정돈된 CSV를 저장합니다.
전처리 명령은 raw 파일명, target, output 경로를 인자로 받아 실행하는 것을 기본으로 합니다.

processed 파일에 반영해도 되는 작업:

- 컬럼명 strip, rename
- 사용하지 않을 컬럼 drop
- target label mapping
- 중복 row 제거
- target이 비어 있는 row 제거
- 명확한 타입 정리와 단위 정리

processed 파일에 미리 반영하지 말아야 하는 작업:

- scaler fit
- encoder fit
- imputer fit
- train/test 전체를 보고 결정한 feature selection
- test set 정보를 이용한 threshold/model 선택

fit이 필요한 변환은 train split 이후 pipeline 안에서 처리합니다.

---

## 7. Data Card 작성 기준

`reports/DATA_CARD.md`에는 최소한 다음을 채웁니다.

- 데이터 출처와 사용 조건
- 실행한 EDA notebook
- row/column 수와 target column
- 주요 feature 설명
- target 분포
- 결측치/이상치/중복 요약
- split 방법과 seed
- 누수 의심 컬럼
- processed 파일 경로와 적용한 전처리 결정
- 편향, 개인정보, 라벨 품질 위험

그래프를 넣는 것보다 “그래프에서 무엇을 판단했는가”가 중요합니다.

---

## 8. 모델링 전 완료 조건

- [ ] target 분포를 확인했다.
- [ ] split 기준과 seed를 정했다.
- [ ] 누수 의심 컬럼을 점검했다.
- [ ] AutoML 전에 제외할 leakage 후보 컬럼을 확정했다.
- [ ] baseline과 AutoML이 같은 `data_version`과 split 조건을 쓰도록 정했다.
- [ ] test set을 최종 확인용으로만 쓰기로 정했다.
- [ ] baseline에 사용할 feature 후보를 정했다.
- [ ] `data/processed/`에 학습용 CSV를 만들었다.
- [ ] fit이 필요한 전처리를 train pipeline으로 남겼다.
- [ ] `data_manifest.json`에 데이터 버전을 기록했다.
- [ ] `reports/DATA_CARD.md` 초안을 작성했다.
