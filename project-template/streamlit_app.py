"""
Streamlit 배포 진입점
학생들이 이 파일을 채워서 배포합니다.

최종 앱 목표:
- model_registry.json 기반 모델 버전별 metric 추이
- experiments/runs/<run_id>/confusion_matrix.json 기반 confusion matrix
- logs/inference.jsonl 기반 요청 수, 에러율, 평균/P95 latency, 최근 로그
"""
import streamlit as st

st.title("ML Project Demo")
st.write("팀명 / 프로젝트명을 여기에 입력하세요.")

user_input = st.text_input("입력값을 넣어주세요")

if st.button("예측하기"):
    # TODO: 모델 로드 및 예측 로직 연결
    st.write("예측 결과가 여기에 표시됩니다.")
    st.info("src/models/ 에서 모델을 로드하고 이 부분을 채워주세요.")
