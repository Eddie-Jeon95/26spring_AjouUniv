# FAQ - 자주 묻는 질문 / 자주 막히는 문제

---

## Git / GitHub

**Q. Git conflict(충돌)이 났어요. 어떻게 해결하나요?**

```bash
# 먼저 최신 main을 가져오기
git pull --rebase origin main

# 충돌 파일을 열어서 <<<<<, =====, >>>>> 표시 부분을 직접 수정
# 수정 후
git add 충돌파일명
git rebase --continue
```

팀원과 같은 파일을 동시에 수정했을 때 주로 발생합니다. 팀원과 직접 소통 후 해결하세요.

---

**Q. API key를 GitHub에 올려버렸어요!**

1. 즉시 해당 서비스에서 key를 무효화(revoke)하고 새 key를 발급받으세요
2. `.gitignore`에 `.env`를 추가하세요
3. git history에서 제거하려면 `git filter-repo` 또는 GitHub 지원팀에 문의

```bash
# .gitignore에 추가
echo ".env" >> .gitignore
echo ".streamlit/secrets.toml" >> .gitignore
```

커밋 전 습관적으로 확인:
```bash
grep -r "sk-" .          # OpenAI key 패턴
grep -r "hf_" .          # HuggingFace token 패턴
```

---

## Python / 패키지

**Q. `pip install -r requirements.txt`가 오류나요.**

```bash
# 가상환경 새로 만들기
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 업그레이드 후 재설치
pip install --upgrade pip
pip install -r requirements.txt
```

버전 충돌이 의심될 때:
```bash
pip install pipreqs
pipreqs . --force   # 실제 import 기반으로 requirements.txt 재생성
```

---

## Streamlit

**Q. Streamlit 앱이 실행은 되는데 모델 로딩이 너무 느려요.**

`@st.cache_resource` 데코레이터를 사용하면 앱 재실행 시 모델을 다시 로드하지 않습니다:

```python
import streamlit as st

@st.cache_resource
def load_model():
    # 모델 로드 코드
    return model

model = load_model()
```

---

**Q. Streamlit Cloud에 배포했는데 앱이 안 뜨거나 크래시가 나요.**

주요 원인:
1. **메모리 초과**: 모델이 1GB를 넘으면 Streamlit Cloud에서 실행 불가. HF Hub에 올리고 런타임에 다운로드하는 방식 사용
2. **requirements.txt 오류**: 로컬에서 `pip install -r requirements.txt` 먼저 테스트
3. **secrets 누락**: `.streamlit/secrets.toml`이 배포 환경에 설정되어 있는지 확인 (Streamlit Cloud 설정 페이지에서 입력)

Streamlit Cloud 로그 확인: 앱 페이지 우측 하단 "Manage app" → "Logs"

---

**Q. 모델 파일(.pkl, .pt 등)을 GitHub에 올릴 수 없어요 (용량 초과).**

HuggingFace Hub를 사용하세요:

```python
# 업로드
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="./models/model.pkl",
    path_in_repo="model.pkl",
    repo_id="YOUR_HF_USERNAME/YOUR_REPO",
    token="hf_..."  # .env에서 불러오기
)

# 다운로드 (앱에서)
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id="YOUR_HF_USERNAME/YOUR_REPO", filename="model.pkl")
```

---

## Colab / 학습 환경

**Q. Colab에서 학습한 모델을 로컬로 가져오려면요?**

```python
# Colab에서 Google Drive에 저장
from google.colab import drive
drive.mount('/content/drive')

import shutil
shutil.copy('/content/model.pkl', '/content/drive/MyDrive/model.pkl')
```

또는 직접 다운로드:
```python
from google.colab import files
files.download('/content/model.pkl')
```

---

**Q. CUDA OOM (Out of Memory) 오류가 나요.**

```python
# batch size 줄이기
# train_loader에서 batch_size 파라미터 감소

# 캐시 비우기
import torch
torch.cuda.empty_cache()

# Mixed precision 사용 (메모리 절약)
from torch.cuda.amp import autocast
with autocast():
    output = model(input)

# CPU로 전환 (학습은 느리지만 메모리 문제 없음)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

---

## Docker

**Q. Docker 컨테이너가 실행 안 돼요.**

```bash
# 로그 확인
docker logs [컨테이너명]
docker logs $(docker ps -lq)   # 가장 최근 컨테이너

# 이미지 재빌드
docker-compose down
docker-compose up --build

# 컨테이너 내부 접속해서 직접 확인
docker exec -it [컨테이너명] /bin/bash
```

---

## 기타

**Q. 팀원 간 Python 버전이 달라서 문제가 생겨요.**

`pyproject.toml`이나 `.python-version` 파일로 버전을 명시하거나, Docker를 사용하면 환경을 통일할 수 있습니다.

**Q. 실험 결과를 어디에 기록해야 하나요?**

`/log-experiment` 커맨드를 사용하면 `experiments/YYYYMMDD_실험명/result.md` 형식으로 자동 안내됩니다. 실패한 실험도 반드시 기록하세요 — 나중에 가장 귀한 자료가 됩니다.
