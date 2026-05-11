# ML Project Specs

이 폴더는 프로젝트의 판단 기준을 모아둔 곳입니다.
commands는 작업 순간의 절차를 안내하고, specs는 무엇을 좋은 결과로 볼지 정의합니다.

---

## 문서별 역할

| 문서 | 언제 읽나 | 핵심 질문 |
|------|-----------|-----------|
| `PROJECT_SPEC.md` | 프로젝트 시작, Streamlit 기준 정리 | 무엇을 풀고 어떤 대시보드를 성공으로 볼 것인가? |
| `DATA_ANALYSIS_SPEC.md` | EDA, split, 데이터 카드 작성 | 데이터가 믿을 만하고 누수가 없는가? |
| `MODELING_SPEC.md` | baseline, 실험 계획, 모델 개선 | 실험이 가설과 비교 기준을 갖고 있는가? |
| `METRICS_AND_INTERPRETATION_SPEC.md` | metric 선택, 모델 비교, error analysis | 숫자를 어떻게 해석하고 설명할 것인가? |

---

## 권장 사용 흐름

1. 프로젝트 시작: `PROJECT_SPEC.md`를 채우며 문제, metric, Streamlit 대시보드 기준을 정합니다.
2. 데이터 확보 후: `DATA_ANALYSIS_SPEC.md`를 기준으로 EDA와 data card를 작성합니다.
3. baseline 전: `MODELING_SPEC.md`를 읽고 가장 단순한 기준 모델을 만듭니다.
4. 실험 후: `METRICS_AND_INTERPRETATION_SPEC.md`를 기준으로 metric과 오류 유형을 해석합니다.
5. Streamlit 정리 전: `model_registry.json`, `experiments/runs/`, `logs/inference.jsonl`이 대시보드에 필요한 정보를 담는지 확인합니다.

---

## Commands와의 관계

- `/plan-experiment`: `PROJECT_SPEC.md`, `MODELING_SPEC.md`, `METRICS_AND_INTERPRETATION_SPEC.md`를 실행 전 체크리스트로 사용합니다.
- `/log-experiment`: `MODELING_SPEC.md`의 실험 기록 기준을 만족했는지 확인합니다.
- `/compare-models`: `METRICS_AND_INTERPRETATION_SPEC.md`의 비교 기준을 따릅니다.
- `/analyze-errors`: `DATA_ANALYSIS_SPEC.md`와 `METRICS_AND_INTERPRETATION_SPEC.md`의 오류 분석 기준을 따릅니다.
