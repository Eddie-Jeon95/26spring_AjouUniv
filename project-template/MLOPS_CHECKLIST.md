# ML 프로젝트 체크리스트

> 이 체크리스트는 평가표가 아니라 프로젝트 품질을 높이기 위한 자가 점검표입니다.
> 데이터 분석, 모델링, 모델 개선 근거, 버전 관리, Streamlit 시각화가 서로 연결되어 있는지 확인합니다.

---

## 1. 문제 정의

- [ ] `docs/specs/PROJECT_SPEC.md`를 확인했는가?
- [ ] 해결하려는 문제가 한 문장으로 명확한가?
- [ ] 입력과 출력이 구체적으로 정의되어 있는가?
- [ ] 예측 오류가 실제 사용자나 서비스에 어떤 영향을 주는지 설명할 수 있는가?
- [ ] 주요 metric과 보조 metric을 정했고, 선택 이유를 설명할 수 있는가?

---

## 2. 데이터 이해와 품질

- [ ] `docs/specs/DATA_ANALYSIS_SPEC.md`의 EDA 기준을 확인했는가?
- [ ] `reports/DATA_CARD.md`에 데이터 출처, 수집 조건, 주요 컬럼을 기록했는가?
- [ ] 결측치, 이상치, 중복, target 분포를 확인했는가?
- [ ] EDA 결과를 바탕으로 `data/processed/`에 학습용 CSV를 만들었는가?
- [ ] train/validation/test split 기준과 seed를 기록했는가?
- [ ] `data_manifest.json`에 데이터 버전, checksum, row/column 수를 남겼는가?
- [ ] 개인정보, 편향, 라벨 오류 가능성을 검토했는가?

---

## 3. 데이터 누수 점검

- [ ] target과 직접 연결된 컬럼이 feature에 포함되지 않았는가?
- [ ] scaler, encoder, imputer 등은 train 데이터로만 fit했는가?
- [ ] 시간 순서가 중요한 문제에서 미래 정보가 섞이지 않았는가?
- [ ] 중복 샘플이 train/test에 동시에 들어가지 않았는가?

---

## 4. Baseline과 실험 설계

- [ ] `docs/specs/MODELING_SPEC.md`의 baseline/실험 기록 기준을 확인했는가?
- [ ] 가장 단순한 baseline을 먼저 만들었는가?
- [ ] baseline metric을 `model_registry.json`에 기록했는가?
- [ ] 새 실험마다 가설, 변경점, 비교 기준을 정했는가?
- [ ] 실험 config와 metric이 `experiments/runs/`에 저장되는가?
- [ ] Streamlit 시각화를 위한 `confusion_matrix.json`이 run마다 저장되는가?
- [ ] 실패한 실험도 기록했는가?

---

## 5. 모델 비교와 개선 근거

- [ ] `docs/specs/METRICS_AND_INTERPRETATION_SPEC.md`의 비교 기준을 확인했는가?
- [ ] 최소 2개 이상의 모델 또는 feature 실험을 비교했는가?
- [ ] metric 개선이 우연인지, 데이터 split이나 seed에 취약한지 확인했는가?
- [ ] 성능이 좋아진 이유를 데이터나 error case로 설명할 수 있는가?
- [ ] `reports/EXPERIMENT_REPORT.md`에 주요 실험 결과를 요약했는가?
- [ ] 최종 선택 모델의 trade-off를 설명할 수 있는가?

---

## 6. Error Analysis

- [ ] `docs/specs/METRICS_AND_INTERPRETATION_SPEC.md`의 error analysis 기준을 확인했는가?
- [ ] 오답 샘플을 직접 확인했는가?
- [ ] 실패 유형을 2~5개 정도로 묶었는가?
- [ ] 특정 class, 구간, 사용자군, 입력 길이에서 성능이 낮은지 확인했는가?
- [ ] `reports/ERROR_ANALYSIS.md`에 원인 가설과 개선 후보를 기록했는가?

---

## 7. 재현성과 버전 관리

- [ ] `CLAUDE.md`의 Claude Code 작업 기준을 확인했는가?
- [ ] `python scripts/train.py --data ... --target ... --data-version ...` 실행 흐름이 유지되는가?
- [ ] seed, config, data_version, model_id가 실험마다 남는가?
- [ ] 큰 데이터와 모델 파일은 Git에 올리지 않고 외부 링크나 경로만 기록했는가?
- [ ] DVC/MLflow를 쓰지 않아도 기본 실험 추적이 가능한가?
- [ ] 심화 도구를 썼다면 사용 이유와 재현 방법을 문서화했는가?

---

## 8. 데모와 간단한 운영 관점

- [ ] Streamlit에서 모델 버전별 metric 추이와 confusion matrix를 확인할 수 있는가?
- [ ] Streamlit에서 추론 요청 수, 에러율, 평균/P95 latency를 확인할 수 있는가?
- [ ] CPU 환경에서도 추론이 동작하는가?
- [ ] 추론 로그에 timestamp, latency, status 정도는 남는가?
- [ ] 모델의 한계와 배포 후 모니터링해야 할 위험을 설명했는가?

---

## 9. 결과 분석과 Streamlit 확인

- [ ] 문제 정의와 metric 선택 이유가 최종 결과 해석에 반영됐는가?
- [ ] EDA 핵심 발견이 전처리나 feature/model 변경으로 이어졌는가?
- [ ] 데이터 품질/누수 점검 결과가 모델 선택 근거에 반영됐는가?
- [ ] baseline과 주요 실험의 차이를 같은 data_version 기준으로 설명할 수 있는가?
- [ ] error analysis 결과가 다음 실험 후보와 연결되는가?
- [ ] 최종 모델의 버전, 한계, 운영 위험이 `MODEL_CARD.md`에 정리됐는가?
- [ ] Streamlit에서 실험 metric, confusion matrix, 추론 로그를 함께 확인할 수 있는가?
