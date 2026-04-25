"""
모델 정의 및 학습/추론 모듈
"""


class BaseModel:
    """모든 모델이 상속받는 기본 클래스"""

    def __init__(self, config: dict):
        """
        Args:
            config: 모델 하이퍼파라미터 및 설정
        """
        self.config = config
        self.model = None

    def build(self):
        """모델 구조를 정의한다."""
        raise NotImplementedError

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        모델을 학습한다.

        Args:
            X_train: 학습 입력
            y_train: 학습 레이블
            X_val:   검증 입력 (선택)
            y_val:   검증 레이블 (선택)
        """
        raise NotImplementedError

    def predict(self, X):
        """
        입력에 대한 예측값을 반환한다.

        Args:
            X: 추론 입력
        Returns:
            예측 결과
        """
        if self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다. train() 또는 load()를 먼저 호출하세요.")
        raise NotImplementedError

    def evaluate(self, X_test, y_test) -> dict:
        """
        테스트 데이터로 성능을 평가하고 메트릭을 반환한다.

        Returns:
            {"accuracy": ..., "f1": ..., ...} 형태의 dict
        """
        raise NotImplementedError

    def save(self, path: str):
        """모델을 파일로 저장한다."""
        raise NotImplementedError

    def load(self, path: str):
        """저장된 모델을 불러온다."""
        raise NotImplementedError
