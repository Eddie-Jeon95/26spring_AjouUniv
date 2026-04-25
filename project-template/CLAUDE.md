# [팀명 / 프로젝트명]

> 🔄 **이 문서는 살아있는 문서입니다.**
> 다음이 바뀌면 반드시 업데이트하세요:
> - 폴더 구조가 추가/변경될 때
> - 새 의존성이나 도구가 추가될 때
> - 팀 내 코딩 규칙이나 컨벤션이 정해질 때
> - 새로운 함정이나 실수가 발견될 때
>
> Claude Code는 이 문서를 매 작업의 기준으로 삼습니다.
> 이 문서가 부정확하면 Claude의 작업도 부정확해집니다.

---

## 1. 프로젝트 개요

### 무엇을 만드는가
- **문제**: [어떤 문제를 해결하나? 한 문장으로]
- **사용자**: [누가 이걸 사용하나?]
- **입력 → 출력**: [예: 영어 에세이 텍스트 → 점수와 피드백]

### 성공 기준
- **ML 메트릭**: [예: F1 score 0.85 이상]
- **운영 메트릭**: [예: 추론 시간 2초 이내]
- **사용자 경험**: [예: Streamlit UI에서 결과를 직관적으로 볼 수 있어야 함]

---

## 2. 기술 스택

- **언어/프레임워크**: Python 3.x
- **학습 환경**: Google Colab (T4 GPU) / 개인 데스크톱
- **배포**: Streamlit Community Cloud
- **모델 호스팅**: [Streamlit Cloud 직접 / Hugging Face Hub - 모델 크기에 따라]
- **실험 기록**: `experiments/` 폴더의 JSON/Markdown 로그
- **운영 로그**: `logs/inference.jsonl` (요청/응답/latency 기록)
- **버전 관리**: GitHub
- **AI 협업 도구**: Claude Code

> Streamlit Community Cloud 제약을 항상 염두에 두세요:
> - RAM 약 1GB, CPU only
> - public repo만 무료
> - secrets는 `.streamlit/secrets.toml`로 관리

---

## 3. 폴더 구조

```
[실제 만든 구조와 일치시킬 것 - 변경 시 이 섹션 업데이트]

src/
├── data/           # 데이터 로드, 전처리
├── models/         # 모델 정의, 학습 코드
├── app/            # Streamlit UI 컴포넌트
└── utils/          # 공통 유틸리티

experiments/        # 실험 기록 (실험별 폴더)
logs/               # 운영 로그 (inference.jsonl 등, .gitignore 권장)
notebooks/          # EDA, 탐색용 (배포 코드 X)
tests/              # 테스트 코드
.streamlit/         # Streamlit 설정
streamlit_app.py    # 배포 진입점
requirements.txt
.env.example
```

---

## 4. 코딩 규칙

### 데이터
- 원본 데이터는 `data/raw/`, 가공 데이터는 `data/processed/` (둘 다 `.gitignore`)
- 데이터 처리 코드는 반드시 재현 가능 (시드 고정, 버전 명시)
- train/val/test 분리 후 절대 섞지 말 것 (데이터 누수 주의)

### 실험
- 모든 실험은 `experiments/YYYYMMDD_실험명/` 폴더에 결과 저장
- 결과는 `result.md`에 기록 (가설, 메트릭, 다음 액션)
- 실패한 실험도 반드시 기록

### 운영 로그
- 추론 요청마다 `logs/inference.jsonl`에 다음 포맷으로 기록:

```json
{
  "timestamp": "2026-04-25T10:30:00",
  "request_id": "abc123",
  "input_summary": "...",
  "prediction": "...",
  "latency_ms": 123,
  "status": "success"
}
```

- `logs/` 폴더는 `.gitignore`에 추가 (용량 및 개인정보 이슈)
- 주기적으로 latency 평균, 에러율, 요청 수를 확인하는 습관 들이기

### Git
- 브랜치: `feature/xxx`, `fix/xxx`, `experiment/xxx`
- 커밋: 한 가지 작업 단위로 작게 자주
- 커밋 메시지: `[타입] 한 줄 요약` (타입: feat / fix / data / model / docs / refactor)
- PR은 `.github/PULL_REQUEST_TEMPLATE.md` 따를 것

### 코드
- secrets는 절대 코드에 하드코딩 금지 (`.env` 또는 `st.secrets` 사용)
- 시드는 항상 고정 (`random`, `numpy`, `torch` 등)
- requirements.txt는 버전 명시 (`==` 권장)

---

## 5. Claude Code 협업 가이드

### ✅ 권장하는 방식
- 작업 시작 전 관련 파일을 먼저 읽도록 요청
- 큰 작업은 단계별로 쪼개서 요청
- 생성된 코드는 직접 실행하고 결과 확인 후 다음 단계로
- 모르는 코드는 설명 요청 후 머지
- 슬래시 커맨드 활용: `/start-feature`, `/log-experiment`, `/log-inference`, `/review-pr`

### ❌ 피해야 할 방식
- "전체 다 만들어줘" 같은 큰 단위 요청
- 검증 없이 바로 커밋
- 에러 메시지를 그대로 던지지 말고, 어디서 발생했는지 파악 후 질문
- 이해 못 하는 코드를 그대로 두기

### 데이터/모델 관련 작업 시 특히 주의
- 데이터 누수 가능성 (test 데이터로 전처리 fit 등)
- 시드 미고정으로 인한 재현 불가
- 평가 메트릭이 문제 정의와 맞지 않는 경우

---

## 6. 현재 진행 상황

> 매주 또는 마일스톤마다 업데이트

### 완료
- [ ] 문제 정의 및 데이터 확보
- [ ] EDA
- [ ] Baseline 모델
- [ ] 본 모델 학습
- [ ] Streamlit UI
- [ ] 배포
- [ ] 모니터링 로깅 연결
- [ ] 발표 준비

### 현재 작업 중
- [채워주세요]

### 다음 마일스톤
- [채워주세요]

---

## 7. 알려진 함정 / 자주 하는 실수

> 아래는 공통적으로 자주 발생하는 문제입니다.
> 프로젝트 진행하며 팀만의 함정을 발견하면 아래에 추가하세요.

### 환경 / 배포
- **Colab 모델 가중치가 1GB 초과** → Streamlit Cloud 배포 실패. HF Hub에 올리고 런타임에 다운로드
- **requirements.txt 버전 충돌** → `pip install pipreqs && pipreqs .`로 자동 생성 후 검토
- **Docker 실행 오류** → `docker logs [컨테이너명]`으로 로그 먼저 확인
- **CPU only 환경에서 GPU 코드 실행 오류** → `device = "cuda" if torch.cuda.is_available() else "cpu"` 패턴 사용

### 데이터 / 모델
- **데이터 누수** → train/test 분리 전에 scaler를 전체 데이터로 fit하는 실수. 반드시 train으로만 fit
- **시드 미고정** → `random.seed()`, `np.random.seed()`, `torch.manual_seed()` 모두 설정
- **CUDA OOM** → batch size 줄이기, `torch.cuda.empty_cache()`, gradient checkpointing 검토

### Git / 협업
- **secrets를 GitHub에 커밋** → `.gitignore`에 `.env`, `secrets.toml` 반드시 추가. 이미 커밋했다면 즉시 key rotation
- **Git conflict** → `git pull --rebase origin main` 습관화, 충돌 파일은 팀원과 직접 소통 후 해결
- **API key 노출** → GitHub Secret Scanning이 자동 감지하지만, 커밋 전 `grep -r "sk-" .`으로 직접 확인

### [팀이 직접 추가할 영역]
- (발견한 함정을 여기에 추가)

---

## 8. 팀 정보

- **팀원**: [이름 / 역할]
- **정기 미팅**: [요일, 시간]
- **커뮤니케이션**: [Slack 채널 / Discord 등]
