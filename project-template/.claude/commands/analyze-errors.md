# /analyze-errors

모델의 실패 케이스를 분석하고 다음 개선 실험을 찾는 커맨드입니다.

---

최근 실험의 예측 결과와 metric을 보고 error analysis를 도와주세요.

## 1단계: 입력 파일 확인

- `docs/specs/DATA_ANALYSIS_SPEC.md`에서 데이터 품질/누수 기준 확인
- `docs/specs/METRICS_AND_INTERPRETATION_SPEC.md`에서 error analysis 기준 확인
- `model_registry.json`에서 분석할 run_id 확인
- 해당 `experiments/runs/<run_id>/metrics.json` 읽기
- 가능하면 `experiments/runs/<run_id>/confusion_matrix.json` 읽기
- 가능하면 `experiments/runs/<run_id>/predictions.csv` 읽기
- 데이터 원본이나 샘플이 없으면, 현재 가능한 범위와 한계를 먼저 말하기

## 2단계: 실패 유형화

```markdown
## Error Types

| 유형 | 관찰 근거 | 가능한 원인 | 확인할 추가 분석 |
|------|-----------|-------------|------------------|
|      |           |             |                  |
```

## 3단계: 개선 후보 제안

- 데이터 품질 개선
- feature engineering
- metric 또는 threshold 조정
- 모델/하이퍼파라미터 변경
- 추가 수집이나 라벨 재검토

## 4단계: 리포트 반영

`reports/ERROR_ANALYSIS.md`에 옮길 수 있는 요약 문장을 제안해주세요.

## 5단계: Spec 기준 점검

- metric 해석 기준을 만족한 점:
- 아직 부족한 분석:
- 다음 실험과 연결되는 개선 후보:

---

주의:
- 개인정보나 민감한 원본 샘플은 그대로 출력하지 마세요.
- 근거 없이 모델 변경만 제안하지 말고, 어떤 실패 유형을 줄이려는지 연결하세요.
