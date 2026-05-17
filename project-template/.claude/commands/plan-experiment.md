---
description: "새 실험을 시작하기 전에 가설, 비교 기준, 실패 조건을 정리합니다."
argument-hint: "[실험 아이디어]"
disable-model-invocation: true
---

# /plan-experiment

새로운 ML 실험을 시작하기 전에 가설, 비교 기준, 실패 조건을 정리하는 커맨드입니다.

---

$ARGUMENTS 실험을 계획하려고 합니다.

바로 코드를 작성하지 말고 다음 순서로 정리해주세요.

## 1단계: 현재 기준선 확인

- `CLAUDE.md`에서 Claude Code 작업 기준과 폴더 규칙 확인
- `docs/specs/PROJECT_SPEC.md`에서 문제 정의와 성공 기준 확인
- `docs/specs/MODELING_SPEC.md`에서 baseline/실험 설계 기준 확인
- `docs/specs/METRICS_AND_INTERPRETATION_SPEC.md`에서 metric 해석 기준 확인
- `data_manifest.json`에서 사용할 데이터 버전 확인
- `model_registry.json`에서 현재 baseline과 최고 성능 실험 확인
- 관련 config와 최근 `experiments/runs/` 결과 확인

## 2단계: 실험 계획 작성

````markdown
# Experiment Plan

## 가설
- 무엇을 바꾸면 왜 좋아질 것이라고 예상하는가?

## 변경점
- 데이터:
- Feature:
- 모델:
- 하이퍼파라미터:

## 비교 기준
- 비교할 baseline model_id:
- 주요 metric:
- 보조 metric:

## 데이터 누수 위험
- 의심되는 지점:
- 방지 방법:

## 성공 / 실패 판단
- 성공 기준:
- 실패해도 배울 수 있는 점:

## 실행 명령
```bash
python scripts/train.py --decisions reports/EXPERIMENT_REPORT.md
```

## 참조한 기준
- CLAUDE.md:
- PROJECT_SPEC:
- MODELING_SPEC:
- METRICS_AND_INTERPRETATION_SPEC:
````

## 3단계: 확인 질문

정보가 부족하면 코드 작성 전에 질문해주세요.

---

주의:
- metric 하나만 보지 말고 error analysis 계획을 함께 제안하세요.
- 새로운 라이브러리는 꼭 필요한 경우에만 제안하세요.
- 학생에게 config YAML 파일 수정을 기본 요구로 두지 말고, 실행 옵션은 `reports/EXPERIMENT_REPORT.md`의 `training_decisions` block에 우선 표현하세요.
