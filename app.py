from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import FinanceDataReader as fdr
import pandas as pd
from datetime import datetime, timedelta

app = FastAPI(title="Stock Chart API", description="주식 차트 조회 API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def index():
    return FileResponse('templates/index.html')

@app.get("/api/stock/{stock_code}")
async def get_stock_data(stock_code: str):
    try:
        # 최근 1년 데이터 가져오기
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # FinanceDataReader로 주식 데이터 가져오기
        df = fdr.DataReader(stock_code, start_date, end_date)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="주식 코드를 찾을 수 없습니다.")
        
        # 데이터 정리
        df = df.reset_index()
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        
        stock_data = {
            'dates': df['Date'].tolist(),
            'prices': {
                'open': df['Open'].tolist(),
                'high': df['High'].tolist(),
                'low': df['Low'].tolist(),
                'close': df['Close'].tolist()
            },
            'volume': df['Volume'].tolist(),
            'stock_code': stock_code
        }
        
        return stock_data
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데이터를 가져오는 중 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)