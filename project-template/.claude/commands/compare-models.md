# /compare-models

여러 실험의 성능과 trade-off를 비교하는 커맨드입니다.

---

`model_registry.json`과 `experiments/runs/`의 결과를 바탕으로 모델을 비교해주세요.

## 1단계: 실험 목록 정리

- `CLAUDE.md`에서 모델 비교 원칙 확인
- `docs/specs/METRICS_AND_INTERPRETATION_SPEC.md`에서 metric 해석 기준 확인
- `model_registry.json`을 읽고 model_id, data_version, metric, artifact path를 확인
- 각 run의 `confusion_matrix.json`이 있으면 주요 혼동 패턴도 확인
- 각 run의 `config.yaml`에서 split, seed, target, feature 구성이 같은지 확인
- 같은 data_version끼리 우선 비교
- split이나 data_version이 다른 실험은 직접 비교가 어렵다고 표시

## 2단계: 비교 표 작성

| model_id | data_version | split/seed | 주요 metric | 보조 metric | 변경점 | 장점 | 한계 |
|----------|--------------|------------|-------------|-------------|--------|------|------|
|          |              |            |             |             |        |      |      |

## 3단계: 해석

```markdown
## 결론
- 현재 가장 설득력 있는 모델:
- 이유:
- 아직 불확실한 점:
- spec 기준에서 부족한 근거:

## 다음 실험 제안
1.
2.
3.
```

## 4단계: 발표에 쓸 문장 제안

metric 숫자 나열이 아니라, 왜 모델이 개선됐는지 설명하는 3~5문장 요약을 제안해주세요.

---

주의:
- data_version이 다르면 그 차이를 반드시 언급하세요.
- 같은 data_version이라도 split, seed, feature 구성이 다르면 직접 우열 비교를 보류하세요.
- accuracy만 좋아지고 특정 class recall이 나빠진 경우 경고하세요.
