# DB 공모전

## 가상환경 설정 방법
```bash
# (1) 가상환경 생성 (한 번만)
python -m venv venv

# (2) 가상환경 활성화
.\venv\Scripts\activate   # Windows
source venv/bin/activate  # MacOS

# (참고) 가상환경 비활성화
deactivate
```

## 패키지 설치 방법
```bash
# 개별 패키지 설치 방법
pip install [패키지명]

# requirements.txt 생성: 현재 환경의 패키지 목록 저장
pip freeze > requirements.txt

# requirements.txt로 설치: 저장된 파일로 패키지 일괄 설치
pip install -r requirements.txt
```