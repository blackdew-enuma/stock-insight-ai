# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

주식 차트 조회 웹 애플리케이션 - FinanceDataReader를 사용하여 주식 데이터를 가져와 웹에서 차트로 표시하는 FastAPI 애플리케이션입니다.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the FastAPI application
python app.py
# or
uvicorn app:app --reload

# The app will be available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

## Project Architecture

- **Backend**: FastAPI web server (app.py)
  - `/api/stock/{stock_code}` - 주식 데이터 API 엔드포인트
  - FinanceDataReader로 최근 1년간 주식 데이터 조회
  - JSON 형태로 가격 및 거래량 데이터 반환
  - 자동 API 문서 생성 (/docs)

- **Frontend**: templates/index.html
  - Chart.js를 사용한 인터랙티브 차트
  - 주가 변동 라인 차트
  - 거래량 바 차트
  - 반응형 디자인

## Stock Code Examples

테스트용 주식 코드:
- 005930: 삼성전자
- 000660: SK하이닉스  
- 035720: 카카오
- 207940: 삼성바이오로직스