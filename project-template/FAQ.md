# FAQ - 자주 묻는 질문 / 자주 막히는 문제

---

## 데이터 / 모델 실험

**Q. 데이터 파일이 아직 없는데 `train.py`를 실행해도 되나요?**

네. 기본 설정에서는 `data/raw/dataset.csv`가 없으면 scikit-learn 샘플 데이터로 baseline 흐름을 확인합니다. 실제 프로젝트를 시작하면 raw 파일을 `data/raw/`에 두고, 전처리 후 `data/processed/` 파일을 만들어 학습하세요.

사용자가 YAML을 직접 수정하는 대신, 실제 프로젝트에서는 보통 아래처럼 실행합니다.

```bash
python scripts/preprocess.py \
  --input data/raw/your_data.csv \
  --output data/processed/your_data_v1.csv \
  --target label \
  --data-version your-data-v1

python scripts/train.py \
  --data data/processed/your_data_v1.csv \
  --target label \
  --data-version your-data-v1
```

---

**Q. 실험 결과를 어디에 기록해야 하나요?**

`python scripts/train.py --config configs/default.yaml`를 실행하면 다음이 생성됩니다.

- `experiments/runs/<run_id>/config.yaml`
- `experiments/runs/<run_id>/metrics.json`
- `experiments/runs/<run_id>/predictions.csv`
- `experiments/runs/<run_id>/confusion_matrix.json`
- `experiments/runs/<run_id>/result.md`
- `model_registry.json`의 새 모델 기록
- `data_manifest.json`의 데이터 버전 기록

실험 해석은 `/log-experiment` 커맨드와 `reports/EXPERIMENT_REPORT.md`를 함께 사용하세요. 실패한 실험도 반드시 기록하세요.

---

**Q. 데이터 버전 관리는 DVC를 꼭 써야 하나요?**

아니요. 기본 템플릿은 `data_manifest.json`, `model_registry.json`, `experiments/runs/`만으로도 재현에 필요한 정보를 남기도록 설계했습니다. 데이터가 크거나 실험 수가 많아지면 DVC나 MLflow를 선택적으로 검토하세요.

---

**Q. 모델 파일은 어디에 두나요?**

기본 학습 스크립트는 `models/`에 `.pkl` 파일을 저장합니다. 이 폴더는 `.gitignore`에 포함되어 있으므로 GitHub에는 올라가지 않습니다. 데모나 재현에 필요한 모델은 Google Drive, Hugging Face Hub 등 외부 링크를 `model_registry.json`에 기록하세요.

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

같은 파일을 동시에 수정했을 때 주로 발생합니다. 충돌 표시가 남지 않도록 수정 후 다시 실행하거나 diff를 확인하세요.

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

**Q. 이 프로젝트는 어떤 Python 버전을 써야 하나요?**

Python **3.10.x**를 기준으로 사용하세요. `project-template/.python-version`에는 `3.10.13`이 적혀 있습니다.
AutoGluon까지 설치하는 A-to-Z 흐름은 Python 3.10 환경을 전제로 합니다.

확인:

```bash
python3.10 --version
```

가상환경 안에서는 아래처럼 확인합니다.

```bash
python --version
```

`Python 3.10.x`가 아니면 가상환경을 다시 만드세요.

---

**Q. `python3.10` 명령이 없어요.**

macOS에서 `pyenv`를 쓴다면:

```bash
pyenv install 3.10.13
pyenv local 3.10.13
python --version
```

Windows에서는 Python 공식 설치 파일을 사용하거나 Python Launcher로 확인합니다.

```powershell
py -0p
py -3.10 --version
```

Python 3.10을 설치한 뒤 가상환경을 다시 만듭니다.

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
```

---

**Q. `requirements.txt`에는 무엇이 들어 있나요?**

`requirements.txt` 하나에 EDA, baseline, Streamlit, AutoGluon, SHAP 의존성을 모두 포함합니다.
이 템플릿에서는 아래 명령만 사용합니다.

```bash
pip install -r requirements.txt
```

---

**Q. `pip install -r requirements.txt`가 오래 걸리거나 실패해요.**

AutoGluon은 여러 tabular 모델 후보와 optional dependency를 설치하므로 시간이 오래 걸릴 수 있습니다. 먼저 아래를 확인하세요.

```bash
python --version
python -m pip --version
```

Python이 3.10.x가 아니면 가상환경을 지우고 다시 만듭니다.

macOS / Linux:

```bash
deactivate  # 활성화되어 있을 때만
rm -rf venv
python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
deactivate
Remove-Item -Recurse -Force venv
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

설치가 끝난 뒤 아래로 확인합니다.

```bash
python -c "from autogluon.tabular import TabularPredictor; print('autogluon OK')"
```

---

**Q. Apple Silicon Mac에서 설치가 잘 안 됩니다.**

먼저 Python 3.10 가상환경인지 확인합니다.

```bash
python --version
```

기존 venv가 다른 Python으로 만들어졌다면 삭제하고 다시 만드세요.

```bash
deactivate
rm -rf venv
python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

AutoGluon 설치가 계속 실패하면 Python 버전과 설치 로그를 먼저 확인합니다. 최종 A-to-Z 제출 전에는 `requirements.txt` 설치와 `autogluon OK` 확인이 필요합니다.

---

**Q. Jupyter Notebook kernel은 어떻게 등록하나요?**

가상환경을 활성화한 상태에서 한 번만 등록합니다.

```bash
python -m ipykernel install --user --name ml-project-venv --display-name "ML Project venv"
```

Notebook에서 kernel을 `"ML Project venv"`로 선택하세요.

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
2. **의존성 오류**: AutoGluon까지 쓰는 프로젝트라면 로컬에서 `pip install -r requirements.txt` 먼저 테스트
3. **secrets 누락**: `.streamlit/secrets.toml`이 배포 환경에 설정되어 있는지 확인 (Streamlit Cloud 설정 페이지에서 입력)

Streamlit Cloud 로그 확인: 앱 페이지 우측 하단 "Manage app" → "Logs"

---

**Q. Streamlit 포트가 이미 사용 중이라고 나와요.**

다른 포트로 실행합니다.

```bash
streamlit run streamlit_app.py --server.port 8502
```

---

**Q. Streamlit 말고 무료로 쓸 수 있는 대안도 있나요?**

기본 템플릿은 Streamlit을 권장합니다. Python만으로 metric, confusion matrix, 추론 로그를 한 화면에 묶기 쉽기 때문입니다.

심화 선택지는 다음 정도를 검토할 수 있습니다.

- **Hugging Face Spaces + Gradio**: 모델 데모 UI에 적합
- **MLflow**: 실험 metric과 artifact 추적에 적합
- **Evidently**: 데이터/모델 모니터링과 drift 분석에 적합

다만 프로젝트 기본 흐름은 Streamlit 기준으로 맞추고, 위 도구들은 필요할 때만 선택적으로 추가하세요.

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

## 기타

**Q. Python 버전이 달라서 문제가 생겨요.**

이 템플릿은 `.python-version`으로 `3.10.13`을 명시합니다.
다른 Python으로 만든 venv를 계속 쓰지 말고 Python 3.10으로 새 venv를 만든 뒤 `pip install -r requirements.txt`를 다시 실행하세요.
