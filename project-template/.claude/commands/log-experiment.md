---
description: "학습 실행 후 실험 결과를 표준 형식으로 기록하고 해석합니다."
argument-hint: "[run_id 선택사항]"
disable-model-invocation: true
---

# /log-experiment

ML 실험 결과를 표준 형식으로 기록하고 해석하는 커맨드입니다.

---

방금 진행한 실험을 `experiments/runs/`, `model_registry.json`, `reports/EXPERIMENT_REPORT.md` 기준으로 정리해주세요.

## 1단계: 실행 결과 확인

- `CLAUDE.md`의 실험 기록 원칙 확인
- `docs/specs/MODELING_SPEC.md`의 실험 기록 기준 확인
- 실행한 config 파일
- 생성된 run_id와 run directory
- `metrics.json`
- `confusion_matrix.json`
- `model_registry.json`에 추가된 record
- 사용한 `data_version`

부족한 정보가 있으면 질문해주세요.

## 2단계: 실험 요약 작성

```markdown
# 실험: [run_id 또는 실험명]

## 가설
- 무엇을 검증하려 했는가?

## 변경점
- 데이터:
- Feature:
- 모델:
- 하이퍼파라미터:

## 결과
| 메트릭 | 값 | 비교 baseline |
|--------|----|---------------|
|        |    |               |

## 해석
- 좋아진 점:
- 나빠진 점:
- 예상과 다른 점:

## 다음 액션
- [ ]
- [ ]
```

## 3단계: 버전 기록 점검

- data_version이 기록됐는가?
- config_hash가 기록됐는가?
- artifact_path가 기록됐는가?
- confusion_matrix_path가 기록됐는가?
- 큰 모델 파일이 Git에 들어가지 않았는가?

## 4단계: Spec 기준 점검

- `MODELING_SPEC.md` 기준을 만족한 점:
- 부족한 점:
- 다음 실험 전에 보강할 점:

## 5단계: 실험에서 배운 점

이 실험에서 배운 점을 결과 분석에 연결되는 한 문장으로 정리해주세요.

---

주의:
- 실패한 실험도 기록하세요.
- metric 개선이 작거나 불안정하면 과장하지 마세요.
