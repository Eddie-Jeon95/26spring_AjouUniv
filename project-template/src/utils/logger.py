"""
추론 로그 기록 모듈
logs/inference.jsonl 에 요청/응답/latency를 저장한다.
"""
import json
import time
import uuid
from datetime import datetime
from pathlib import Path


class InferenceLogger:
    """추론 요청을 JSONL 형식으로 기록하는 클래스"""

    def __init__(self, log_path: str = "logs/inference.jsonl"):
        """
        Args:
            log_path: 로그 파일 경로 (기본: logs/inference.jsonl)
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, input_summary: str, prediction, latency_ms: float, status: str = "success", **kwargs):
        """
        추론 결과를 한 줄 JSON으로 기록한다.

        Args:
            input_summary: 입력 요약 (개인정보 포함 금지)
            prediction:    모델 예측 결과
            latency_ms:    추론 소요 시간 (밀리초)
            status:        "success" 또는 "error"
            **kwargs:      추가로 기록할 필드
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "request_id": str(uuid.uuid4())[:8],
            "input_summary": input_summary,
            "prediction": str(prediction),
            "latency_ms": round(latency_ms, 2),
            "status": status,
            **kwargs,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_error(self, input_summary: str, error: Exception, latency_ms: float):
        """에러 발생 시 로그를 기록한다."""
        self.log(
            input_summary=input_summary,
            prediction=None,
            latency_ms=latency_ms,
            status="error",
            error_type=type(error).__name__,
            error_message=str(error),
        )


# 사용 예시
# logger = InferenceLogger()
#
# start = time.time()
# result = model.predict(user_input)
# elapsed = (time.time() - start) * 1000
#
# logger.log(
#     input_summary=user_input[:100],
#     prediction=result,
#     latency_ms=elapsed,
# )
