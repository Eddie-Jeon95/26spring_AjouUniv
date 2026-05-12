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

## 사용 방식

전체 진행 순서는 프로젝트 루트의 `README.md`를 따릅니다.
이 폴더의 문서들은 각 단계에서 판단 기준이 흔들리지 않게 잡는 체크포인트입니다.

- EDA와 Data Card 작성 전에는 `DATA_ANALYSIS_SPEC.md`를 봅니다.
- baseline과 새 실험 전에는 `MODELING_SPEC.md`를 봅니다.
- metric 해석, 모델 비교, error analysis 전에는 `METRICS_AND_INTERPRETATION_SPEC.md`를 봅니다.
- Streamlit 최종 화면 기준은 `PROJECT_SPEC.md`의 대시보드 항목을 따릅니다.
