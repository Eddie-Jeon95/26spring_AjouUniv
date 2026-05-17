# 26spring AjouUniv — ML 프로젝트 템플릿

아주대학교 2026년 봄학기 인공지능 프로젝트 수업에서 사용할 ML 프로젝트 템플릿 관리 레포입니다.

핵심 산출물은 `project-template/`입니다. 템플릿은 문제 정의, EDA, 전처리, baseline, AutoML, 모델 비교, Streamlit 데모까지 한 흐름으로 진행할 수 있도록 구성되어 있습니다.

---

## 이 레포의 역할

루트 레포는 템플릿의 목적과 구성을 설명합니다.
실제 프로젝트 진행 순서와 실행 명령은 `project-template/README.md`를 기준으로 봅니다.

핵심 설계는 학생과 Claude Code가 같은 폴더 규칙과 판단 기준을 보고 움직이게 하는 것입니다.
코드를 대신 완성하는 것보다, 데이터 분석과 실험 해석 과정에서 판단 근거를 남기는 습관을 잡는 데 초점을 둡니다.

---

## 구성

```text
project-template/        # 복사해서 사용하는 프로젝트 템플릿
├── README.md           # 학생용 quickstart
├── CLAUDE.md           # Claude Code 작업 기준
├── PROJECT_GUIDE.md    # 문제 정의, EDA, 모델링, metric, AutoML, 평가 기준
├── .claude/commands/   # 7단계 slash command
├── .streamlit/         # Streamlit 설정 위치
├── configs/            # 기본 학습 설정
├── data/               # raw/processed 데이터 위치, Git 제외
├── experiments/runs/   # 실행별 metric, prediction, confusion matrix, Git 제외
├── logs/               # 추론 로그, Git 제외
├── models/             # 모델 artifact, Git 제외
├── notebooks/          # EDA와 탐색 분석 notebook
├── reports/PROJECT_REPORT.md # Data Card, Experiment Report, Error Analysis, Model Card
├── scripts/            # 전처리와 학습 진입점
├── src/                # 데이터/모델/로깅 확장 지점
├── requirements.txt    # 전체 Python 의존성
└── streamlit_app.py    # 실험 결과와 추론 로그 시각화 앱
```

---

## 권장 진행

학생이 외워야 할 단계는 아래 7개입니다.

```text
/project
/data
/train
/automl
/evaluate
/demo
/checkpoint
```

긴 Python 명령어를 직접 조립하기보다 `project-template/reports/PROJECT_REPORT.md`의 decision block을 채우고 아래 명령을 사용합니다.

```bash
python scripts/preprocess.py --decisions reports/PROJECT_REPORT.md
python scripts/train.py --decisions reports/PROJECT_REPORT.md
python scripts/train_automl.py --decisions reports/PROJECT_REPORT.md
```

---

## 배포 기준

학생에게 배포할 대상은 `project-template/`입니다.

Git 기준 배포에서는 `.gitignore`에 잡힌 raw/processed 데이터, 모델 파일, 로그, 실행 산출물이 포함되지 않습니다.
폴더를 직접 복사하거나 zip으로 공유할 때는 로컬 검증 산출물이 섞이지 않았는지 확인하세요.
