"""
모델 학습 실행 스크립트

사용법:
    python scripts/train.py --config configs/default.yaml
"""
import argparse
import random
import numpy as np


def set_seed(seed: int = 42):
    """재현 가능한 학습을 위해 시드를 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    # PyTorch 사용 시 주석 해제
    # import torch
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    """YAML 설정 파일을 불러온다."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def main(config: dict):
    """
    전체 학습 파이프라인을 실행한다.

    순서:
        1. 데이터 로드 및 전처리
        2. 모델 정의
        3. 학습
        4. 평가
        5. 모델 저장
        6. 실험 결과 기록 (experiments/ 폴더)
    """
    set_seed(config.get("seed", 42))

    # TODO: 1. 데이터 로드
    # from src.data.preprocessing import DataPreprocessor
    # preprocessor = DataPreprocessor(config["data"])
    # df = preprocessor.load(config["data"]["path"])
    # ...

    # TODO: 2. 모델 정의
    # from src.models.model import BaseModel
    # model = BaseModel(config["model"])
    # model.build()

    # TODO: 3. 학습
    # model.train(X_train, y_train, X_val, y_val)

    # TODO: 4. 평가
    # metrics = model.evaluate(X_test, y_test)
    # print(metrics)

    # TODO: 5. 모델 저장
    # model.save(config["model"]["save_path"])

    # TODO: 6. 실험 기록 → /log-experiment 커맨드 활용


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
