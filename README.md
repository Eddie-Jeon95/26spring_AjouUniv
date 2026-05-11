# 26spring AjouUniv — ML 프로젝트 템플릿

아주대학교 2026년 봄학기 인공지능 프로젝트 수업을 위한 조교 준비 자료입니다.

---

## 이 저장소의 목적

학생들이 데이터 분석, baseline 모델링, 모델 개선, 실험 비교, 간단한 배포/모니터링까지 경험할 수 있도록 돕는 템플릿입니다.

핵심은 코드를 대신 완성하는 것이 아니라, 학생들이 스스로 더 좋은 ML 프로젝트를 만들 수 있도록 **실험 구조와 기록 습관**을 미리 잡아주는 것입니다.

---

## 구성

```
project-template/        ← 학생들이 복사해서 사용하는 템플릿
├── CLAUDE.md            ← Claude Code 작업 기준과 폴더 규칙
├── MLOPS_CHECKLIST.md   ← ML 품질 중심 자가 점검 체크리스트
├── AI_GUIDE.md          ← EDA, feature, error analysis 중심 AI 활용 가이드
├── FAQ.md               ← 자주 막히는 문제 모음
├── README.md            ← 학생 프로젝트 시작점
│
├── .claude/commands/    ← Claude Code 슬래시 커맨드
│   ├── plan-experiment  ← 실험 전 가설과 비교 기준 정리
│   ├── log-experiment   ← 실험 결과 표준 형식으로 기록
│   ├── compare-models   ← 여러 모델/실험 결과 비교
│   ├── analyze-errors   ← 실패 케이스 분석
│   └── log-inference    ← 간단한 추론 로그 점검
│
├── notebooks/           ← EDA 템플릿
├── docs/specs/          ← 문제 정의, 데이터 분석, 모델링, 지표 해석 기준
├── reports/             ← 데이터 카드, 실험 리포트, 오류 분석, 모델 카드
├── scripts/train.py     ← baseline 학습과 실험 기록 진입점
├── streamlit_app.py     ← 실험 결과와 추론 로그 시각화 진입점
├── configs/             ← 실험 설정 파일
├── data_manifest.json   ← 데이터 버전 기록
└── model_registry.json  ← 모델 버전 기록
```

---

## 학생 사용 흐름

| 시점 | 액션 |
|------|------|
| 프로젝트 시작 | 템플릿 복사 → `project-template/README.md`의 처음 시작하는 순서 확인 |
| 문제 정의 | `docs/specs/PROJECT_SPEC.md`에 입력/출력, metric, Streamlit 기준 작성 |
| 데이터 분석 | `notebooks/01_eda_template.ipynb`로 EDA 수행 후 `reports/DATA_CARD.md` 작성 |
| baseline 확인 | `python scripts/train.py --config configs/default.yaml` 실행 |
| 새 실험 전 | `/plan-experiment [실험 아이디어]` |
| 실험 후 | `/log-experiment` |
| 여러 실험 비교 | `/compare-models` |
| 성능 개선 고민 | `/analyze-errors` |
| 데모 / 로그 확인 | Streamlit에서 모델별 metric 추이, confusion matrix, 추론 로그 확인 |
| 데모 후 | `/log-inference` |

---

## 학습 목표

- 데이터 이해와 품질 점검
- baseline 모델 수립
- 실험 가설 기반 모델 개선
- 데이터 버전과 모델 버전 기록
- Claude Code를 활용한 분석과 개발 생산성 향상
- Streamlit 기반 실험 결과와 추론 로그 시각화

---

## 학생 시작 방법

1. `project-template/` 폴더를 새 GitHub 레포로 복사
2. `project-template/README.md`의 `0. 처음 시작하는 순서` 확인
3. 가상환경 생성 후 의존성 설치
4. `python scripts/train.py --config configs/default.yaml`로 baseline 흐름 확인
5. `docs/specs/`, `reports/DATA_CARD.md`, `reports/EXPERIMENT_REPORT.md`를 프로젝트에 맞게 채우기

```bash
cd project-template
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/train.py --config configs/default.yaml
```
