# Model Card

## 1. 모델 개요

- **최종 model_id**:
- **run_id**:
- **data_version**:
- **모델 종류**:
- **experiment type**: baseline / automl
- **AutoML backend**: AutoGluon Tabular 또는 해당 없음
- **artifact 위치**:
- **leaderboard 위치**:
- **automl_summary 위치**:

## 2. 사용 목적

- 해결하려는 문제:
- 입력:
- 출력:
- 사용하면 좋은 상황:
- 사용하면 안 되는 상황:

## 3. 성능

| 데이터 split | 주요 metric | 보조 metric | 비고 |
|--------------|-------------|-------------|------|
| validation   |             |             |      |
| test         |             |             |      |

- **primary metric 선택 이유**:
- **threshold 선택 기준**:
- **baseline 대비 개선/악화**:

## 4. 학습 데이터

- 데이터 출처:
- 데이터 버전:
- 주요 전처리:
- 알려진 편향 / 한계:

## 5. 한계와 위험

- 성능이 낮은 입력 유형:
- 데이터 누수 방지 확인:
- AutoGluon pipeline 복잡도 / 운영 비용:
- test set을 모델 선택에 사용하지 않았는지:
- 배포 후 모니터링해야 할 점:
- 윤리적 / 개인정보 이슈:

## 6. 재현 방법

```bash
python scripts/train.py --config configs/default.yaml
```

AutoGluon 모델을 최종 후보로 선택했다면:

```bash
python scripts/train_automl.py --config configs/default.yaml
```

## 7. 향후 개선 방향

- [ ]
- [ ]
