# Project Spec

프로젝트가 무엇을 예측하고, 어떤 기준을 성공으로 볼지 짧게 정의합니다.
이 문서는 문제 정의와 최종 Streamlit 대시보드 기준을 흔들리지 않게 잡는 용도입니다.

---

## 1. 문제 정의

다음 문장을 채울 수 있어야 합니다.

```markdown
이 프로젝트는 [사용자/상황]에서 [입력 데이터]를 바탕으로 [출력]을 예측하여 [의사결정/문제]를 돕는다.
가장 중요한 오류 비용은 [false positive / false negative / 큰 오차]이며, 성공 기준은 [주요 metric과 목표]이다.
```

확인할 질문:

- 어떤 현상이나 의사결정을 예측/분류/추천하려는가?
- 입력은 무엇이고 출력은 무엇인가?
- 모델 결과를 누가, 어떤 상황에서 사용할 것인가?
- 틀렸을 때 어떤 비용이 생기는가?
- ML이 단순 규칙이나 통계 요약보다 왜 필요한가?

---

## 2. 입력과 출력

- **입력 데이터**:
- **예측 대상 / target**:
- **예측 결과 형식**: class label / score / probability / numeric value
- **예측 시점에 알 수 없는 정보**:

예측 시점에 알 수 없는 정보가 feature에 들어가면 data leakage입니다.

---

## 3. 사용자와 사용 상황

- **사용자**:
- **사용자가 모델 결과를 보고 할 행동**:
- **틀렸을 때 가장 큰 위험**:
- **Streamlit 데모에서 넣을 입력 예시**:

사용 상황이 불명확하면 metric 선택과 오류 해석도 흔들립니다.

---

## 4. 주요 Metric과 성공 기준

| 기준 | 프로젝트 기준 |
|------|-----------|
| 주요 metric | |
| 보조 metric | |
| baseline 대비 목표 | |
| 반드시 줄이고 싶은 오류 유형 | |
| 데이터/모델 버전 기록 기준 | `data_manifest.json`, `model_registry.json`, `experiments/runs/` 기록 |

목표 수치가 아직 없으면 baseline을 먼저 만든 뒤 “baseline 대비 개선”으로 정의합니다.

---

## 5. 최종 Streamlit 대시보드 기준

최종 결과는 단순 예측 화면만이 아니라, 실험 결과와 추론 로그를 함께 보여주는 Streamlit 화면입니다.

최소 포함 항목:

- 모델 버전별 주요 metric 추이: `model_registry.json`의 `created_at`, `model_id`, `data_version`, `metrics`
- 선택한 run의 confusion matrix: `experiments/runs/<run_id>/confusion_matrix.json`
- 최근 추론 로그 테이블: `logs/inference.jsonl`
- 추론 요청 수, 성공/실패 요청 수, 에러율
- 평균 latency와 P95 latency
- 모델 입력/출력 예시 또는 요약

이 템플릿에서 “학습 추이”는 epoch별 loss curve가 아니라, 실험/run/model version별 metric 변화를 뜻합니다.

### 추론 로그 최소 스키마

`logs/inference.jsonl`은 한 줄에 하나의 JSON object를 기록합니다.
Streamlit 대시보드는 최소한 다음 필드를 읽을 수 있어야 합니다.

| 필드 | 의미 |
|------|------|
| `timestamp` | 추론 요청 시각 |
| `model_id` | 사용한 모델 버전 |
| `run_id` | 연결된 실험 run |
| `status` | `success` 또는 `error` |
| `latency_ms` | 요청 처리 시간 |
| `input_summary` | 개인정보를 제거한 입력 요약 |
| `prediction` | 예측 결과 또는 요약 |
| `error_message` | 실패 시 에러 요약, 성공 시 빈 값 |

로그에는 이름, 이메일, 학번, 원문 민감정보를 그대로 남기지 않습니다.
