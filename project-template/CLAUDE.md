# Claude Code Guide

> 이 문서는 Claude Code가 이 프로젝트에서 따라야 할 단일 작업 기준입니다.
> 프로젝트 판단 기준은 이 문서와 `docs/specs/`를 함께 따릅니다.

---

## 1. 프로젝트 목적

이 템플릿의 목표는 학생들이 ML 프로젝트를 다음 흐름으로 끝까지 수행하도록 돕는 것입니다.

1. 문제 정의
2. 데이터 분석과 품질 점검
3. baseline 모델 구축
4. 실험 가설 기반 모델 개선
5. 모델/데이터 버전 기록
6. 결과 해석과 error analysis
7. Streamlit 기반 실험 결과와 추론 로그 시각화

코드를 대신 완성하는 것보다, 학생이 판단 근거를 남기며 개선하도록 돕는 것이 우선입니다.

---

## 2. 먼저 읽을 문서

Claude Code로 작업을 시작할 때는 다음 순서로 컨텍스트를 잡습니다.

1. `CLAUDE.md`: Claude Code 작업 기준과 폴더 규칙
2. `docs/specs/README.md`: spec 문서 사용법
3. `docs/specs/PROJECT_SPEC.md`: 문제 정의, 성공 기준, Streamlit 대시보드 기준
4. `docs/specs/DATA_ANALYSIS_SPEC.md`: EDA와 데이터 품질 기준
5. `docs/specs/MODELING_SPEC.md`: baseline과 실험 기록 기준
6. `docs/specs/METRICS_AND_INTERPRETATION_SPEC.md`: metric 해석과 error analysis 기준
7. `MLOPS_CHECKLIST.md`: 진행 중 자가 점검
8. `AI_GUIDE.md`: Claude에게 분석/실험을 요청하는 방식

작업 요청이 애매하면 위 spec을 기준으로 부족한 정보를 먼저 확인하고, 추측으로 코드를 확장하지 않습니다.

---

## 3. 폴더와 산출물 규칙

```
configs/              # 학습 설정 YAML
data/raw/             # 원본 데이터, Git 제외
data/processed/       # 가공 데이터, Git 제외
docs/specs/           # 프로젝트 판단 기준 문서
experiments/runs/     # 실행별 config, metrics, predictions, confusion matrix, result
notebooks/            # EDA와 탐색 분석
reports/              # 데이터/실험/오류/모델 해석 리포트
scripts/train.py      # baseline 학습과 실험 기록 진입점
src/data/             # 전처리와 데이터 로드 확장 지점
src/models/           # 모델 정의와 학습/추론 확장 지점
src/utils/            # 로깅 등 공통 유틸리티
streamlit_app.py      # 실험 결과와 추론 로그를 보여주는 앱 진입점
logs/                 # 추론 로그, Git 제외
```

작업 단계별로 주로 쓰는 위치:

| 단계 | 주로 보는 파일/폴더 |
|------|---------------------|
| 데이터 분석 | `data/raw/`, `notebooks/`, `reports/DATA_CARD.md`, `data_manifest.json` |
| 모델링 | `configs/`, `scripts/train.py`, `src/data/`, `src/models/`, `experiments/runs/`, `model_registry.json` |
| 결과 해석 | `experiments/runs/<run_id>/metrics.json`, `predictions.csv`, `confusion_matrix.json`, `reports/EXPERIMENT_REPORT.md`, `reports/ERROR_ANALYSIS.md` |
| 추가 모델링 / 데이터 추가 | 새 config, 새 `data_version`, 새 run 기록, `model_registry.json` |
| Streamlit 시각화 | `streamlit_app.py`, `src/utils/logger.py`, `logs/inference.jsonl`, `model_registry.json`, `experiments/runs/` |

핵심 기록 파일:

- `data_manifest.json`: 데이터 버전, 출처, 크기, checksum, split 기록
- `model_registry.json`: model_id, data_version, config hash, metric, artifact 경로 기록
- `experiments/runs/<run_id>/confusion_matrix.json`: Streamlit에서 선택한 run의 confusion matrix 시각화에 사용
- `logs/inference.jsonl`: Streamlit 데모 추론 요청, 상태, latency 기록
- `reports/DATA_CARD.md`: 데이터 설명과 품질 위험
- `reports/EXPERIMENT_REPORT.md`: 주요 실험의 가설, 결과, 해석
- `reports/ERROR_ANALYSIS.md`: 실패 유형과 개선 후보
- `reports/MODEL_CARD.md`: 최종 모델 사용 조건과 한계

---

## 4. 작업 원칙

### 데이터

- 원본 데이터와 가공 데이터는 Git에 올리지 않습니다.
- 데이터가 바뀌면 `data_manifest.json`의 `data_version`, row/column 수, split, checksum을 갱신합니다.
- train/validation/test split 이후에는 test 데이터를 전처리 fit이나 모델 선택에 사용하지 않습니다.
- leakage가 의심되면 모델 성능 개선보다 누수 확인을 먼저 합니다.

### 모델링

- 복잡한 모델보다 단순한 baseline을 먼저 만듭니다.
- 새 실험은 반드시 가설, 변경점, 비교 baseline, 성공/실패 기준을 가집니다.
- config, metric, prediction sample, artifact path가 재현 가능하게 남아야 합니다.
- 실험별 `confusion_matrix.json`을 남겨 Streamlit에서 모델별 오류 패턴을 확인할 수 있어야 합니다.
- 모델 파일은 `models/` 또는 외부 저장소에 두고 Git에 올리지 않습니다.

### 결과 해석

- metric 숫자만 보고 결론을 내리지 않습니다.
- data_version이나 split이 다른 실험은 직접 비교하지 않습니다.
- class imbalance, threshold, error type, 모델 한계를 함께 봅니다.
- 최종 Streamlit 화면에서는 모델 버전별 metric 추이, 선택한 run의 confusion matrix, 추론 로그의 요청 수/에러율/latency를 함께 확인할 수 있어야 합니다.

### Claude 사용

- Claude가 만든 코드는 학생이 실행하고 이해한 뒤 반영합니다.
- API key, 개인정보, 민감한 원본 샘플은 프롬프트에 넣지 않습니다.
- 새 라이브러리는 꼭 필요한 경우에만 제안하고 `requirements.txt`를 함께 갱신합니다.
- 큰 작업은 계획 → 구현 → 실행 확인 → 문서 갱신 순서로 진행합니다.
- 결과가 좋아 보여도 data leakage, split 변경, metric 선택 문제를 먼저 점검합니다.

---

## 5. Claude Code 권장 커맨드

| 커맨드 | 언제 사용 | 참조 기준 |
|--------|-----------|-----------|
| `/plan-experiment` | 새 feature/model 실험 전 | `CLAUDE.md`, `MODELING_SPEC.md`, `METRICS_AND_INTERPRETATION_SPEC.md` |
| `/log-experiment` | 학습 실행 후 | `CLAUDE.md`, `MODELING_SPEC.md` |
| `/compare-models` | 여러 run 비교 | `CLAUDE.md`, `METRICS_AND_INTERPRETATION_SPEC.md` |
| `/analyze-errors` | 오답 원인 분석 | `DATA_ANALYSIS_SPEC.md`, `METRICS_AND_INTERPRETATION_SPEC.md` |
| `/log-inference` | 데모/배포 후 로그 점검 | `CLAUDE.md`, `PROJECT_SPEC.md` |

---

## 6. Claude에게 요청할 때 포함할 정보

```markdown
## Goal
[이번 작업의 목표]

## Project Context
- 문제 정의:
- data_version:
- baseline model_id:
- 주요 metric:

## Relevant Specs
- CLAUDE.md
- docs/specs/[관련 문서]

## Task
[구체적인 요청]

## Expected Output
- 변경/분석 요약
- 확인한 기준
- 실행 또는 검증 방법
```

---

## 7. 현재 진행 상황

### 완료

- [ ] 문제 정의
- [ ] 데이터 확보
- [ ] EDA와 데이터 카드 작성
- [ ] 데이터 버전 기록
- [ ] Baseline 모델
- [ ] 실험 로그와 모델 registry 기록
- [ ] Error analysis
- [ ] 모델 개선 실험
- [ ] Streamlit 실험/로그 대시보드
- [ ] 최종 모델 카드
- [ ] 발표 준비

### 현재 작업 중

- [채워주세요]

### 다음 마일스톤

- [채워주세요]

---

## 8. 금지 사항

- 데이터 파일, 모델 가중치, `.env`, Streamlit secrets를 Git에 추가하지 않습니다.
- test set 성능을 반복적으로 보고 모델 선택에 사용하지 않습니다.
- 실패한 실험을 삭제하거나 숨기지 않습니다.
- metric 개선 근거 없이 “성능 향상”이라고 쓰지 않습니다.
- spec 문서와 registry를 갱신하지 않은 채 최종 모델만 남기지 않습니다.
- 새로 발견한 함정이나 팀 규칙은 `CLAUDE.md` 또는 관련 spec에 반영합니다.
