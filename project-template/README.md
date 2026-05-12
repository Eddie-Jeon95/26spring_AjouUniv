# Project Title

> 한 줄로 프로젝트를 설명하세요.

---

## 0. 처음 시작하는 순서

이 README는 학생 프로젝트의 시작점입니다. 처음에는 아래 순서대로 진행하세요.

1. `docs/specs/PROJECT_SPEC.md`에 문제 정의, 입력/출력, 주요 metric, Streamlit 최종 화면 기준을 간단히 적습니다.
2. 원본 데이터를 `data/raw/`에 넣고 Claude에게 파일명과 target 후보를 알려 EDA를 요청합니다.
   - 예: `/eda data/raw/banknote.txt target=label`
3. Claude가 만든 EDA notebook 결과를 보고 `reports/DATA_CARD.md`에 데이터 품질과 전처리 결정을 정리합니다.
4. Claude에게 raw 파일명, target, 저장할 processed 파일명을 알려 전처리를 요청합니다.
   - 예: `/preprocess-data data/raw/banknote.txt target=label output=data/processed/banknote_v1.csv`
5. Claude에게 processed 파일명과 target을 알려 baseline 학습을 요청합니다.
   - 예: `/train-baseline data/processed/banknote_v1.csv target=label data_version=banknote-v1 test_size=0.2 val_size=0.2`
6. 새 실험 전에는 `/plan-experiment [실험 아이디어]`로 가설과 비교 기준을 정합니다.
7. 실험 후에는 `/log-experiment`, `/compare-models`, `/analyze-errors`로 결과와 한계를 기록합니다.
8. 단계가 끝날 때마다 `/checkpoint [메시지]`로 문서/코드 변경을 검토하고 commit합니다.
9. 마지막에 `/check-streamlit`으로 산출물 연결을 점검하고 앱 URL을 확인합니다.

학생이 `configs/default.yaml`을 직접 수정하는 것이 기본 흐름은 아닙니다.
파일명, target, data_version을 Claude Code 대화나 slash command 인자로 알려주면 Claude가 실행 명령을 맞춰줍니다.
새로 추가한 slash command가 현재 Claude Code 세션에서 바로 보이지 않으면, 프로젝트 루트에서 새 세션을 열거나 해당 command 내용을 그대로 붙여 넣어 요청하세요.

먼저 볼 기준 문서:

- `CLAUDE.md`: Claude Code 작업 기준과 폴더 규칙
- `docs/specs/PROJECT_SPEC.md`: 문제 정의, 성공 기준, Streamlit 대시보드 기준
- `docs/specs/DATA_ANALYSIS_SPEC.md`: 데이터 분석과 누수 점검 기준
- `docs/specs/MODELING_SPEC.md`: baseline과 실험 기록 기준
- `docs/specs/METRICS_AND_INTERPRETATION_SPEC.md`: metric 해석과 error analysis 기준

---

## 1. 프로젝트 개요

- **문제 정의**:
- **입력 → 출력**:
- **사용자 / 활용 상황**:
- **주요 평가 지표**:

문제 정의와 성공 기준은 `docs/specs/PROJECT_SPEC.md`에 먼저 정리합니다.
Claude Code와 협업할 때의 작업 기준은 `CLAUDE.md`를 따릅니다.

---

## 2. 팀원 및 역할

| 이름 | 역할 | 담당 내용 |
|------|------|-----------|
|      |      |           |

---

## 3. 데이터 분석

- **데이터 출처**:
- **데이터 버전**: `data_manifest.json`의 `data_version`
- **데이터 크기**:
- **주요 컬럼 / 입력**:
- **EDA 핵심 발견**:
- **품질 이슈 / 편향 가능성**:
- **데이터 누수 점검 결과**:

자세한 내용은 `reports/DATA_CARD.md`에 기록합니다.
분석 기준은 `docs/specs/DATA_ANALYSIS_SPEC.md`를 따릅니다.

---

## 4. 모델링과 실험

- **Baseline 모델**:
- **평가 지표 선택 이유**:
- **주요 실험 요약**:
- **최종 모델 선택 이유**:

| model_id | data_version | 주요 변경점 | metric | 한계 |
|----------|--------------|-------------|--------|------|
|          |              |             |        |      |

자세한 실험 로그는 `experiments/runs/`와 `model_registry.json`을 확인합니다.
Streamlit에서 볼 confusion matrix는 각 run의 `confusion_matrix.json`을 사용합니다.
모델링 기준은 `docs/specs/MODELING_SPEC.md`를 따릅니다.

---

## 5. 모델 개선 근거

- **오류 유형**:
- **개선한 부분**:
- **개선되지 않은 부분**:
- **다음에 시도할 것**:

자세한 내용은 `reports/ERROR_ANALYSIS.md`와 `reports/EXPERIMENT_REPORT.md`에 기록합니다.
metric 해석과 모델 비교 기준은 `docs/specs/METRICS_AND_INTERPRETATION_SPEC.md`를 따릅니다.

---

## 6. 실행 방법

### 환경 설정

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 학습

```bash
python scripts/train.py \
  --data data/processed/banknote_v1.csv \
  --target label \
  --data-version banknote-v1 \
  --test-size 0.2 \
  --val-size 0.2
```

데이터 파일이 아직 없으면 scikit-learn 샘플 데이터로 baseline 흐름을 확인합니다.
실제 프로젝트에서는 먼저 `scripts/preprocess.py`로 `data/processed/`에 학습용 CSV를 만든 뒤 위 명령처럼 실행합니다.

### 전처리

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

전처리 단계에서는 컬럼명 정리, rename/drop, target mapping, 중복 제거, target 결측 제거처럼 파일 자체를 정돈하는 작업만 합니다.
scaler, encoder, imputer처럼 train 데이터에 fit해야 하는 변환은 학습 pipeline에 남깁니다.

### 추론 / 앱 실행

```text
/check-streamlit
```

직접 실행할 때는 아래 명령을 사용합니다.

```bash
streamlit run streamlit_app.py
```

---

## 7. 버전 관리

- **데이터 버전 기록**: `data_manifest.json`
- **모델 버전 기록**: `model_registry.json`
- **실험 산출물**: `experiments/runs/<run_id>/`
- **큰 파일 보관 위치**: [Google Drive / Hugging Face Hub / Kaggle / 기타 링크]

큰 원본 데이터, 가공 데이터, 모델 가중치는 Git에 직접 올리지 않습니다.
작업 단계가 끝나면 `/checkpoint [메시지]`로 `*.md`, 코드, `data_manifest.json`, `model_registry.json` 같은 가벼운 기록 파일만 검토 후 commit합니다.
Claude Code와 협업할 때의 공통 기준은 `CLAUDE.md`를 확인합니다.

---

## 8. 데모와 모니터링

- **데모 방식**: Streamlit Community Cloud / 로컬 실행 / 기타
- **데모 URL**:
- **기록하는 로그**: `logs/inference.jsonl`
- **실험 대시보드**: 모델 버전별 metric 추이, 선택한 run의 confusion matrix
- **로그 대시보드**: 요청 수, error rate, 평균/P95 latency, 최근 추론 로그
- **운영 한계**:

---

## 9. 결과 및 한계

- **최종 결과**:
- **잘 된 점**:
- **한계**:
- **개선 방향**:

---

## 10. AI 도구 사용 내역

| 항목 | 내용 |
|------|------|
| 사용한 도구 | Claude Code |
| 사용 목적 | EDA 질문 생성, feature 후보 검토, 오류 분석 보조 등 |
| AI가 제안한 내용 | |
| 사람이 검토하고 수정한 내용 | |

---

## 11. 참고 자료

- [링크]
