from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

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
        
        # 기본 주식 데이터
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
        
        # 차트 분석 추가
        analysis = generate_chart_analysis(df)
        stock_data['analysis'] = analysis
        
        # 예측 데이터 추가
        prediction = generate_stock_prediction(df)
        stock_data['prediction'] = prediction
        
        return stock_data
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데이터를 가져오는 중 오류가 발생했습니다: {str(e)}")

def generate_chart_analysis(df):
    """차트 분석 결과 생성"""
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    
    # 기본 통계
    price_change = latest['Close'] - prev['Close']
    price_change_pct = (price_change / prev['Close']) * 100
    
    # 이동평균
    df_temp = df.copy()
    df_temp['Close'] = pd.to_numeric(df_temp['Close'])
    ma5 = df_temp['Close'].tail(5).mean()
    ma20 = df_temp['Close'].tail(20).mean()
    ma60 = df_temp['Close'].tail(60).mean() if len(df_temp) >= 60 else ma20
    
    # 거래량 분석
    avg_volume = df_temp['Volume'].mean()
    latest_volume = latest['Volume']
    volume_ratio = latest_volume / avg_volume
    
    # RSI (간단 버전)
    price_changes = df_temp['Close'].diff().dropna()
    gains = price_changes.where(price_changes > 0, 0)
    losses = -price_changes.where(price_changes < 0, 0)
    avg_gain = gains.tail(14).mean()
    avg_loss = losses.tail(14).mean()
    rsi = 100 - (100 / (1 + (avg_gain / avg_loss))) if avg_loss != 0 else 50
    
    analysis = {
        'current_price': float(latest['Close']),
        'price_change': float(price_change),
        'price_change_pct': float(price_change_pct),
        'moving_averages': {
            'ma5': float(ma5),
            'ma20': float(ma20),
            'ma60': float(ma60)
        },
        'volume_analysis': {
            'current_volume': int(latest_volume),
            'avg_volume': int(avg_volume),
            'volume_ratio': float(volume_ratio)
        },
        'technical_indicators': {
            'rsi': float(rsi)
        },
        'trend_analysis': get_trend_analysis(df_temp, ma5, ma20),
        'support_resistance': get_support_resistance(df_temp)
    }
    
    return analysis

def get_trend_analysis(df, ma5, ma20):
    """추세 분석"""
    current_price = df['Close'].iloc[-1]
    
    if current_price > ma5 > ma20:
        trend = "상승추세"
        strength = "강함"
    elif current_price > ma20 and ma5 > ma20:
        trend = "상승추세"
        strength = "보통"
    elif current_price < ma5 < ma20:
        trend = "하락추세"
        strength = "강함"
    elif current_price < ma20 and ma5 < ma20:
        trend = "하락추세"
        strength = "보통"
    else:
        trend = "횡보"
        strength = "약함"
    
    return {"trend": trend, "strength": strength}

def get_support_resistance(df):
    """지지선/저항선 계산"""
    highs = df['High'].tail(20)
    lows = df['Low'].tail(20)
    
    resistance = float(highs.max())
    support = float(lows.min())
    
    return {"support": support, "resistance": resistance}

def generate_stock_prediction(df):
    """주식 예측 생성 (간단한 모델)"""
    now = datetime.now()
    market_close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    # 오늘 장마감 전후 확인
    is_after_close = now > market_close_time
    target_date = "내일" if is_after_close else "오늘"
    
    latest = df.iloc[-1]
    prev_close = float(latest['Close'])
    
    # 최근 5일 변동성 계산
    recent_prices = pd.to_numeric(df['Close'].tail(5))
    volatility = recent_prices.std() / recent_prices.mean()
    
    # 간단한 예측 로직 (실제로는 더 복잡한 ML 모델 사용)
    # 최근 추세와 변동성을 고려한 예측
    recent_trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
    
    # 예측 확률 (랜덤 요소 포함)
    base_prob = 0.5 + (recent_trend * 0.3)  # 추세 반영
    up_probability = max(0.2, min(0.8, base_prob + random.uniform(-0.1, 0.1)))
    
    direction = "상승" if up_probability > 0.5 else "하락"
    confidence = abs(up_probability - 0.5) * 2
    
    # 예측 가격 범위
    price_change_range = prev_close * volatility * random.uniform(0.5, 2.0)
    
    if direction == "상승":
        predicted_open = prev_close + random.uniform(-price_change_range*0.3, price_change_range*0.5)
        predicted_close = prev_close + random.uniform(price_change_range*0.2, price_change_range*0.8)
        predicted_high = max(predicted_open, predicted_close) + random.uniform(0, price_change_range*0.3)
        predicted_low = min(predicted_open, predicted_close) - random.uniform(0, price_change_range*0.2)
    else:
        predicted_open = prev_close - random.uniform(-price_change_range*0.3, price_change_range*0.5)
        predicted_close = prev_close - random.uniform(price_change_range*0.2, price_change_range*0.8)
        predicted_high = max(predicted_open, predicted_close) + random.uniform(0, price_change_range*0.2)
        predicted_low = min(predicted_open, predicted_close) - random.uniform(0, price_change_range*0.3)
    
    # 거래량 예측
    avg_volume = df['Volume'].tail(10).mean()
    predicted_volume = int(avg_volume * random.uniform(0.7, 1.5))
    
    prediction = {
        'target_date': target_date,
        'direction': direction,
        'probability': round(up_probability * 100, 1),
        'confidence': round(confidence * 100, 1),
        'predicted_prices': {
            'open': round(predicted_open, 0),
            'close': round(predicted_close, 0),
            'high': round(predicted_high, 0),
            'low': round(predicted_low, 0)
        },
        'predicted_volume': predicted_volume,
        'current_price': prev_close
    }
    
    return prediction

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)