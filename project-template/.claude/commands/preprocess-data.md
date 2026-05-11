# /preprocess-data

EDA 결과와 Data Card를 참고해 raw 데이터를 학습용 processed CSV로 정리하는 커맨드입니다.

---

입력 인자 예시:

```text
/preprocess-data data/raw/banknote.txt target=label output=data/processed/banknote_v1.csv
```

$ARGUMENTS

## 1단계: 현재 근거 확인

- `CLAUDE.md`의 데이터/전처리 원칙 확인
- `docs/specs/DATA_ANALYSIS_SPEC.md`의 processed 데이터 기준 확인
- `reports/DATA_CARD.md`의 EDA 결과와 전처리 후보 확인
- 인자로 받은 raw 파일, target, output 경로 확인
- header, separator, column names가 불확실하면 raw 파일 첫 줄을 보고 학생에게 확인 질문

## 2단계: 전처리 결정 정리

실행 전에 다음을 짧게 정리하세요.

```markdown
## Preprocess Decisions
- raw input:
- processed output:
- target column:
- separator / header:
- column names:
- rename:
- drop columns:
- target mapping:
- duplicate 처리:
- target missing 처리:
- train pipeline에 남길 fit 기반 변환:
```

## 3단계: preprocess 실행

`scripts/preprocess.py`를 CLI 인자로 실행합니다.

예:

```bash
python scripts/preprocess.py \
  --input data/raw/banknote.txt \
  --output data/processed/banknote_v1.csv \
  --target label \
  --sep "," \
  --header none \
  --columns variance,skewness,curtosis,entropy,label \
  --data-version banknote-v1
```

지원되는 선택 인자:

- `--drop-columns col1,col2`
- `--rename old=new,old2=new2`
- `--target-map old=new,old2=new2`
- `--keep-duplicate-rows`
- `--keep-rows-missing-target`

## 4단계: 기록 갱신

- `data_manifest.json`에 raw/processed path, checksum, row/column 수, data_version이 기록됐는지 확인
- `reports/DATA_CARD.md`에 processed 파일 경로와 전처리 결정을 반영
- 다음 baseline 명령을 제안

```bash
python scripts/train.py --data data/processed/banknote_v1.csv --target label --data-version banknote-v1
```

---

주의:
- scaler, encoder, imputer처럼 fit이 필요한 변환은 여기서 하지 않습니다.
- test set 정보나 성능을 보고 전처리 결정을 하지 않습니다.
- raw/processed 데이터 파일은 Git에 추가하지 않습니다.
