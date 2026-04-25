"""
데이터 로드 및 전처리 모듈
"""
import pandas as pd
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """데이터 로드, 정제, train/val/test 분리를 담당하는 클래스"""

    def __init__(self, config: dict):
        """
        Args:
            config: 전처리 설정 (경로, split 비율, 시드 등)
        """
        self.config = config
        self.fitted = False

    def load(self, path: str) -> pd.DataFrame:
        """raw 데이터를 불러온다."""
        raise NotImplementedError

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """결측치 처리, 이상치 제거 등 기본 정제를 수행한다."""
        raise NotImplementedError

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        train 데이터 기준으로 scaler/encoder를 fit하고 변환한다.
        ⚠️ 반드시 train 데이터에만 호출할 것 (데이터 누수 주의)
        """
        raise NotImplementedError

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        fit된 scaler/encoder로 val/test 데이터를 변환한다.
        fit_transform() 호출 이후에만 사용 가능.
        """
        if not self.fitted:
            raise RuntimeError("fit_transform()을 먼저 호출하세요.")
        raise NotImplementedError

    def split(self, df: pd.DataFrame):
        """
        데이터를 train / val / test로 분리한다.

        Returns:
            (df_train, df_val, df_test)
        """
        raise NotImplementedError
