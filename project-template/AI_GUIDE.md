# Claude Code 활용 가이드

> Claude Code의 목적은 코드를 대신 맡기는 것이 아니라, 데이터와 모델을 더 꼼꼼하게 검토하도록 돕는 것입니다.
> 항상 실행 결과와 근거를 사람이 확인하세요.

---

## 1. 좋은 활용 방향

AI에게 요청하기 전에 먼저 `CLAUDE.md`와 `docs/specs/`의 관련 기준을 확인하세요.
commands는 작업 절차를 돕고, specs는 판단 기준을 제공합니다.

### EDA 관점 넓히기

AI에게 다음을 요청하세요.

```markdown
이 데이터셋의 컬럼 설명, 결측치/이상치 요약, target 분포를 바탕으로
추가로 확인해야 할 EDA 질문 10개를 제안해주세요.
데이터 누수 가능성이 있는 컬럼도 따로 의심 목록으로 정리해주세요.
```

### Feature 후보 만들기

```markdown
현재 문제 정의와 EDA 결과를 바탕으로 feature engineering 후보를 제안해주세요.
각 feature마다 기대 효과, 데이터 누수 위험, 구현 난이도를 표로 정리해주세요.
```

### Metric 해석 검토받기

```markdown
아래 실험 결과에서 metric이 실제 문제 정의와 잘 맞는지 검토해주세요.
accuracy만 볼 때 놓칠 수 있는 위험과 추가로 봐야 할 지표를 제안해주세요.
```

### Error analysis 정리하기

```markdown
오답 샘플과 예측 확률을 보고 실패 유형을 묶어주세요.
각 유형별 원인 가설과 다음 실험 아이디어를 제안해주세요.
```

---

## 2. 슬래시 커맨드 활용

| 커맨드 | 용도 | 언제 사용 |
|--------|------|-----------|
| `/plan-experiment [실험 아이디어]` | 실험 가설과 비교 기준 정리 | 새 모델/feature를 시도하기 전 |
| `/log-experiment` | 실험 결과 기록 | 학습 실행 후 |
| `/compare-models` | 여러 실험 결과 비교 | baseline 이후 성능 비교 시 |
| `/analyze-errors` | 실패 케이스 분석 | metric만으로 이유가 안 보일 때 |
| `/log-inference` | 간단한 추론 로그 점검 | 데모/배포 후 |

각 커맨드는 다음 spec을 함께 읽는 것을 전제로 합니다.

- EDA와 데이터 카드: `docs/specs/DATA_ANALYSIS_SPEC.md`
- 실험 계획과 모델 개선: `docs/specs/MODELING_SPEC.md`
- metric 해석과 오류 분석: `docs/specs/METRICS_AND_INTERPRETATION_SPEC.md`

---

## 3. 요청 템플릿

```markdown
## Goal
[이번 분석/실험의 목표를 한 문장으로]

## Current Evidence
- 데이터 버전: [data_manifest.json의 data_version]
- 현재 baseline: [model_registry.json의 model_id / 주요 metric]
- 관찰한 문제: [예: 특정 class recall이 낮음]
- 참고 spec: [docs/specs/...]

## Constraints
- 데이터 누수 가능성을 우선 확인해주세요.
- 새 라이브러리는 꼭 필요한 경우에만 제안해주세요.
- metric 개선뿐 아니라 해석 가능성과 재현성도 함께 봐주세요.

## Task
[EDA 관점 제안 / feature 후보 / error analysis / metric 해석 등]

## Expected Output
- 확인해야 할 질문
- 실험 가설
- 구현 전 주의점
- 성공/실패 판단 기준
```

---

## 4. 반드시 사람이 확인할 것

- train/validation/test split 이후 전처리 fit이 train에만 적용됐는가?
- target 또는 미래 정보를 담은 컬럼이 feature에 섞이지 않았는가?
- metric이 문제의 비용 구조와 맞는가?
- seed, config, data version이 실험 로그에 남았는가?
- AI가 만든 해석이 실제 데이터 분포와 맞는가?

---

## 5. README에 남길 AI 사용 내역

```markdown
## AI Tool Usage

| 항목 | 내용 |
|------|------|
| 사용한 도구 | Claude Code |
| 사용 목적 | EDA 질문 생성, feature 후보 검토, 오류 분석 보조 |
| AI가 제안한 내용 | [예: 누수 의심 컬럼 목록, threshold 조정 실험] |
| 사람이 검토하고 수정한 내용 | [예: 실제 데이터 확인 후 제외한 제안, 채택한 실험] |
```
