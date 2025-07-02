# 주식 차트 조회 웹 애플리케이션

FinanceDataReader 라이브러리를 사용하여 주식 코드를 입력받고 해당 주식의 가격 변동 차트와 거래량 차트를 웹 페이지에 표시하는 애플리케이션입니다.

## 기능

- 주식 코드 입력 (예: 005930 - 삼성전자)
- 주가와 거래량 통합 차트
- **OpenAI 기반 AI 분석 및 예측**
  - 기술적 분석 및 시장 심리 분석
  - 상승/하락 예측 및 예상 가격
  - 예측 이유 및 상세 근거 제공
  - 신뢰도 요인 및 리스크 분석
- 반응형 웹 디자인

## 설치 및 실행

### UV 사용 (권장)
```bash
# 의존성 설치
uv sync

# 환경변수 설정 (OpenAI API 키 필요)
cp .env.example .env
# .env 파일에서 OPENAI_API_KEY 설정

# 애플리케이션 실행
uv run python app.py
```

### 기존 방법
```bash
# 의존성 설치
pip install -r requirements.txt

# 애플리케이션 실행
python app.py
# 또는
uvicorn app:app --reload
```

웹 브라우저에서 `http://localhost:8000` 접속

## 사용법

1. 웹 페이지에서 주식 코드를 입력합니다 (예: 005930, 000660, 035720)
2. "조회" 버튼을 클릭하거나 Enter 키를 누릅니다
3. 주가 차트와 거래량 차트가 표시됩니다

## 주요 주식 코드 예시

- 005930: 삼성전자
- 000660: SK하이닉스
- 035720: 카카오
- 207940: 삼성바이오로직스
- 068270: 셀트리온

## 기술 스택

- **백엔드**: FastAPI, FinanceDataReader
- **프론트엔드**: HTML, JavaScript, Chart.js
- **데이터**: 한국 주식시장 데이터

## API 문서

FastAPI 실행 후 `http://localhost:8000/docs`에서 자동 생성된 API 문서를 확인할 수 있습니다.