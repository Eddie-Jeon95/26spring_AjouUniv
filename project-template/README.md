# Project Title

> 한 줄로 프로젝트를 설명하세요.

---

## 0. 처음 시작하는 순서

이 README는 학생 프로젝트의 시작점입니다. 처음에는 아래 순서대로 진행하세요.

1. `docs/specs/PROJECT_SPEC.md`에 문제 정의, 입력/출력, 주요 metric, Streamlit 최종 화면 기준을 적습니다.
2. `notebooks/01_eda_template.ipynb`로 EDA를 수행하고 `reports/DATA_CARD.md`에 데이터 품질과 위험을 정리합니다.
3. `data_manifest.json`에 데이터 버전, split, checksum을 기록합니다.
4. `python scripts/train.py --config configs/default.yaml`로 baseline 흐름을 확인합니다.
5. 새 실험 전에는 `/plan-experiment [실험 아이디어]`로 가설과 비교 기준을 정합니다.
6. 실험 후에는 `/log-experiment`, `/compare-models`, `/analyze-errors`로 결과와 한계를 기록합니다.
7. `streamlit run streamlit_app.py`로 모델 버전별 metric, confusion matrix, 추론 로그를 확인합니다.

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
python scripts/train.py --config configs/default.yaml
```

데이터 파일이 아직 없으면 scikit-learn 샘플 데이터로 baseline 흐름을 확인합니다.

### 추론 / 앱 실행

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
