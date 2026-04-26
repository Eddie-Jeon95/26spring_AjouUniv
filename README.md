# 26spring AjouUniv — MLOps 프로젝트 수업 자료

아주대학교 2026년 봄학기 MLOps 실습 수업을 위한 조교 준비 자료입니다.

---

## 이 저장소의 목적

학생들이 데이터 처리부터 모델 학습, 배포, 모니터링까지 **MLOps 전 과정을 End-to-End로 경험**할 수 있도록 돕기 위한 세팅입니다.

코드를 대신 짜주는 게 아니라, **스스로 완성할 수 있는 구조와 흐름을 미리 잡아주는 것**이 핵심입니다. 학생들은 이 템플릿을 복사한 뒤, 문서를 채우고 뼈대 코드를 구현하면서 프로젝트를 완성합니다.

---

## 구성

```
project-template/        ← 학생들이 복사해서 사용하는 GitHub 템플릿
├── CLAUDE.md            ← 프로젝트 가이드 (Claude Code 기준 문서)
├── MLOPS_CHECKLIST.md   ← 단계별 자가 점검 체크리스트
├── AI_GUIDE.md          ← Claude/AI 코딩 도구 활용 가이드
├── FAQ.md               ← 자주 막히는 문제 모음
├── README.md            ← 프로젝트 README 템플릿
│
├── .claude/commands/    ← Claude Code 슬래시 커맨드
│   ├── start-feature    ← 새 기능 구현 전 단계별 계획 수립
│   ├── log-experiment   ← 실험 결과 표준 형식으로 기록
│   ├── log-inference    ← 운영 로그 분석 및 모니터링 점검
│   └── review-pr        ← PR 올리기 전 셀프 리뷰
│
├── .github/             ← PR 템플릿, GitHub Actions 워크플로우
├── src/                 ← 클래스 뼈대 코드 (구현은 학생이)
│   ├── data/preprocessing.py
│   ├── models/model.py
│   └── utils/logger.py
├── scripts/train.py     ← 학습 파이프라인 진입점
└── configs/             ← 실험별 설정 파일
```

---

## 학생 사용 흐름

| 시점 | 액션 |
|------|------|
| 프로젝트 시작 | 템플릿 복사 → `CLAUDE.md` 1·2·3·8번 섹션 채우기 |
| 새 기능 시작 | `/start-feature 구현할 기능명` |
| 실험 후 | `/log-experiment` |
| 배포 후 주기적으로 | `/log-inference` |
| PR 올리기 전 | `/review-pr` |
| 프로젝트 진행 중 | `CLAUDE.md` 4·6·7번 섹션 계속 업데이트 |

---

## 학습 목표

- GitHub 기반 코드 관리 및 팀 협업 경험
- Claude / AI 도구를 활용한 개발 생산성 향상
- 데이터 처리 → 모델링 → 배포 → 모니터링 전 과정 수행
- 기말 프로젝트를 통한 실전 문제 해결 및 발표

---

## 학생 시작 방법

1. 이 저장소의 `project-template/` 폴더를 새 GitHub 레포로 복사
2. `CLAUDE.md`의 **1번(프로젝트 개요), 2번(기술 스택), 3번(폴더 구조), 8번(팀 정보)** 섹션을 채우고 커밋
3. Claude Code를 열고 작업 시작
