from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
from openai import OpenAI

app = FastAPI(title="Stock Chart API", description="주식 차트 조회 API")

# OpenAI 클라이언트 초기화
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

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


@app.get("/api/stock/{stock_code:path}")
async def get_stock_data(stock_code: str):
    """전체 주식 데이터 조회 (AI 분석 제외) - 주식 및 통화 쌍 지원"""
    try:
        # 통화 쌍의 '|'를 '/'로 복원 (프론트엔드에서 URL 라우팅 문제 해결을 위해 대체했음)
        decoded_stock_code = stock_code.replace('|', '/')
        
        # 최대 5년 데이터 가져오기
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*5)
        
        # FinanceDataReader로 데이터 가져오기 (주식, 통화 쌍, 암호화폐 모두 지원)
        try:
            df = fdr.DataReader(decoded_stock_code, start_date, end_date)
        except Exception as e:
            print(f"First attempt failed: {e}")
            # 실패 시 Yahoo Finance로 재시도
            try:
                df = fdr.DataReader(decoded_stock_code, start_date, end_date, source='yahoo')
            except Exception as e2:
                print(f"Yahoo attempt also failed: {e2}")
                raise HTTPException(status_code=404, detail=f"데이터를 찾을 수 없습니다: {decoded_stock_code}. {str(e)}")
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"데이터를 찾을 수 없습니다: {decoded_stock_code}")
        
        # 데이터 정리 - 인덱스를 Date 컬럼으로 변환
        df = df.reset_index()
        
        # 인덱스가 Date로 이름이 없는 경우 첫 번째 컬럼을 Date로 처리
        if df.columns[0] != 'Date':
            df = df.rename(columns={df.columns[0]: 'Date'})
        
        # Date 컬럼을 문자열로 변환
        if 'Date' in df.columns:
            # 이미 datetime 객체인 경우와 그렇지 않은 경우 모두 처리
            try:
                if hasattr(df['Date'].iloc[0], 'strftime'):
                    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
                else:
                    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            except Exception as e:
                print(f"Date conversion error: {e}")
                # 문자열로 변환 시도
                df['Date'] = df['Date'].astype(str)
        
        # 통화 쌍인지 주식인지 구분
        is_currency_pair = '/' in decoded_stock_code
        
        # NaN, Infinity 값 처리 (JSON 직렬화 에러 방지)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        # 안전한 숫자 변환 함수
        def safe_float(x):
            try:
                val = float(x)
                return val if np.isfinite(val) else 0.0
            except (ValueError, TypeError):
                return 0.0
                
        def safe_int(x):
            try:
                val = int(float(x))
                return val if np.isfinite(val) else 0
            except (ValueError, TypeError):
                return 0
        
        # 데이터 반환
        stock_data = {
            'dates': df['Date'].tolist(),
            'prices': {
                'open': [safe_float(x) for x in df['Open']],
                'high': [safe_float(x) for x in df['High']],
                'low': [safe_float(x) for x in df['Low']],
                'close': [safe_float(x) for x in df['Close']]
            },
            'volume': [safe_int(x) for x in df['Volume']] if 'Volume' in df.columns else [0] * len(df),
            'stock_code': decoded_stock_code,
            'is_currency_pair': is_currency_pair
        }
        
        return stock_data
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데이터를 가져오는 중 오류가 발생했습니다: {str(e)}")

@app.get("/api/analysis/{stock_code:path}")
async def get_ai_analysis_only(stock_code: str):
    """AI 분석만 별도로 제공하는 엔드포인트 - 주식 및 통화 쌍 지원"""
    try:
        # 통화 쌍의 '|'를 '/'로 복원
        decoded_stock_code = stock_code.replace('|', '/')
        
        # 최근 1년 데이터로 분석 (AI 분석용)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        df = fdr.DataReader(decoded_stock_code, start_date, end_date)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"데이터를 찾을 수 없습니다: {decoded_stock_code}")
        
        # 기본 차트 분석
        basic_analysis = generate_chart_analysis(df)
        
        # OpenAI를 이용한 상세 분석 및 예측
        if client:
            try:
                ai_analysis = await get_ai_analysis(df, decoded_stock_code, basic_analysis)
                ai_prediction = await get_ai_prediction(df, decoded_stock_code, basic_analysis)
                
                return {
                    'analysis': {**basic_analysis, **ai_analysis},
                    'prediction': ai_prediction
                }
            except Exception as e:
                # AI 분석 실패 시 기본 분석 사용
                prediction = generate_stock_prediction(df)
                return {
                    'analysis': basic_analysis,
                    'prediction': prediction
                }
        else:
            # OpenAI 클라이언트가 없으면 기본 분석만 사용
            prediction = generate_stock_prediction(df)
            return {
                'analysis': basic_analysis,
                'prediction': prediction
            }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류가 발생했습니다: {str(e)}")

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
    
    # 최근 10일 변동성 계산 (더 안정적)
    recent_prices = pd.to_numeric(df['Close'].tail(10))
    price_returns = recent_prices.pct_change().dropna()
    volatility = price_returns.std()
    
    # 변동성이 너무 작거나 NaN인 경우 기본값 사용
    if pd.isna(volatility) or volatility < 0.01:
        volatility = 0.02  # 기본 변동성 2%
    
    # 변동성이 너무 큰 경우 제한
    volatility = min(volatility, 0.1)  # 최대 10% 변동성
    
    # 간단한 예측 로직 (실제로는 더 복잡한 ML 모델 사용)
    # 최근 추세와 변동성을 고려한 예측
    if len(recent_prices) >= 2:
        recent_trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
    else:
        recent_trend = 0
    
    # 예측 확률 (랜덤 요소 포함)
    base_prob = 0.5 + (recent_trend * 0.3)  # 추세 반영
    up_probability = max(0.2, min(0.8, base_prob + random.uniform(-0.1, 0.1)))
    
    direction = "상승" if up_probability > 0.5 else "하락"
    confidence = abs(up_probability - 0.5) * 2
    
    # 예측 가격 범위 (현재가 기준 퍼센트로 계산)
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
    
    # 가격이 음수가 되지 않도록 보정
    predicted_open = max(predicted_open, prev_close * 0.1)  # 최소 현재가의 10%
    predicted_close = max(predicted_close, prev_close * 0.1)
    predicted_high = max(predicted_high, max(predicted_open, predicted_close))
    predicted_low = max(predicted_low, prev_close * 0.05)  # 최소 현재가의 5%
    
    # 논리적 일관성 확인
    predicted_high = max(predicted_high, predicted_open, predicted_close)
    predicted_low = min(predicted_low, predicted_open, predicted_close)
    
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

async def get_ai_analysis(df, stock_code, basic_analysis):
    """OpenAI를 이용한 상세 주식 분석"""
    if not client:
        raise Exception("OpenAI client not available")
    try:
        # 최근 30일 데이터 준비
        recent_data = df.tail(30)
        
        # 데이터 요약 생성
        data_summary = {
            'recent_prices': recent_data['Close'].tolist()[-10:],
            'recent_volumes': recent_data['Volume'].tolist()[-10:],
            'current_price': float(basic_analysis['current_price']),
            'price_change_pct': float(basic_analysis['price_change_pct']),
            'rsi': float(basic_analysis['technical_indicators']['rsi']),
            'ma5': float(basic_analysis['moving_averages']['ma5']),
            'ma20': float(basic_analysis['moving_averages']['ma20']),
            'trend': basic_analysis['trend_analysis']['trend'],
            'support': float(basic_analysis['support_resistance']['support']),
            'resistance': float(basic_analysis['support_resistance']['resistance'])
        }
        
        # 통화 쌍인지 주식인지 구분하여 적절한 분석 컨텍스트 제공
        asset_type = "환율" if "/" in stock_code else "주식"
        
        prompt = f"""
        {stock_code}의 {asset_type} 기술적 분석을 수행해주세요. 
        현재 시장 동향과 최신 경제 뉴스를 고려하여 전문적인 분석을 제공해주세요.
        
        현재 데이터:
        - 현재가: {data_summary['current_price']:,}{'원' if asset_type == '주식' else ''}
        - 전일 대비: {data_summary['price_change_pct']:.2f}%
        - RSI: {data_summary['rsi']:.1f}
        - 5일 이평선: {data_summary['ma5']:,}
        - 20일 이평선: {data_summary['ma20']:,}
        - 추세: {data_summary['trend']}
        - 지지선: {data_summary['support']:,}
        - 저항선: {data_summary['resistance']:,}
        - 최근 10일 종가: {data_summary['recent_prices']}
        - 최근 10일 거래량: {data_summary['recent_volumes']}
        
        분석 시 고려사항:
        - 현재 전 세계 경제 상황과 금융시장 동향
        - 해당 {asset_type}에 영향을 미치는 주요 경제 지표
        - 최근 뉴스와 시장 이벤트
        - 기술적 지표의 신뢰성과 시장 심리
        
        다음 형식으로 JSON 응답해주세요:
        {{
            "market_sentiment": "강세/약세/중립",
            "key_insights": ["주요 인사이트 1", "주요 인사이트 2", "주요 인사이트 3"],
            "technical_summary": "기술적 분석 요약 (100자 이내)",
            "risk_factors": ["리스크 요인 1", "리스크 요인 2"],
            "detailed_analysis": "상세한 차트 분석 및 해석 (500자 이내)"
        }}
        """
        
        # 웹서치 기능을 활용한 실시간 분석
        analysis_input = f"""
        {stock_code}의 {asset_type} 기술적 분석을 수행해주세요. 
        최신 뉴스, 시장 동향, 경제 지표를 웹에서 검색하여 실시간 정보를 반영한 전문적인 분석을 제공해주세요.
        
        현재 데이터:
        - 현재가: {data_summary['current_price']:,}{'원' if asset_type == '주식' else ''}
        - 전일 대비: {data_summary['price_change_pct']:.2f}%
        - RSI: {data_summary['rsi']:.1f}
        - 5일 이평선: {data_summary['ma5']:,}
        - 20일 이평선: {data_summary['ma20']:,}
        - 추세: {data_summary['trend']}
        - 지지선: {data_summary['support']:,}
        - 저항선: {data_summary['resistance']:,}
        
        반드시 다음과 같은 올바른 JSON 형식으로만 응답해주세요. 다른 텍스트는 포함하지 마세요:
        {{
            "market_sentiment": "강세/약세/중립",
            "key_insights": ["주요 인사이트 1", "주요 인사이트 2", "주요 인사이트 3"],
            "technical_summary": "기술적 분석 요약 (100자 이내)",
            "risk_factors": ["리스크 요인 1", "리스크 요인 2"],
            "detailed_analysis": "상세한 차트 분석 및 해석 (500자 이내)"
        }}
        """
        
        # 분석 프롬프트 출력
        print(f"\n🔍 [AI 분석 프롬프트] for {stock_code}")
        print("=" * 80)
        print(analysis_input)
        print("=" * 80)
        
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": f"당신은 경험이 풍부한 {asset_type} 전문 분석가입니다. 현재 시장 상황과 최신 경제 동향을 바탕으로 정확하고 전문적인 분석을 제공해주세요."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": analysis_input
                        }
                    ]
                }
            ],
            response_format={
                "type": "json_object"
            },
            temperature=0.7,
            max_completion_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        import json
        # chat.completions API의 응답 형식에 맞게 수정
        response_content = response.choices[0].message.content
        
        # GPT 응답 출력
        print(f"\n📊 [AI 분석 응답] for {stock_code}")
        print("=" * 80)
        print(response_content)
        print("=" * 80)
        
        try:
            ai_analysis = json.loads(response_content)
            return ai_analysis
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}")
            print("기본 분석으로 fallback")
            # JSON 파싱 실패 시 기본 분석 반환
            raise Exception("AI 분석 JSON 파싱 실패")
        
    except Exception as e:
        print(f"AI 분석 오류: {e}")
        return {
            "market_sentiment": "중립",
            "key_insights": ["AI 분석을 사용할 수 없습니다"],
            "technical_summary": "기본 기술적 분석만 제공됩니다",
            "risk_factors": ["AI 분석 오류"],
            "detailed_analysis": "OpenAI API를 통한 상세 분석을 사용할 수 없습니다"
        }

async def get_ai_prediction(df, stock_code, basic_analysis):
    """OpenAI를 이용한 주식 예측"""
    if not client:
        raise Exception("OpenAI client not available")
    try:
        now = datetime.now()
        market_close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
        is_after_close = now > market_close_time
        target_date = "내일" if is_after_close else "오늘"
        
        # 최근 데이터 준비
        recent_data = df.tail(20)
        current_price = float(basic_analysis['current_price'])
        
        asset_type = "환율" if "/" in stock_code else "주식"
        price_unit = "" if asset_type == "환율" else "원"
        
        prompt = f"""
        {stock_code}의 {target_date} {asset_type} 가격을 예측해주세요.
        최신 경제 동향, 시장 상황, 정치적 요인들을 종합적으로 고려하여 전문적인 예측을 제공해주세요.
        
        현재 상황:
        - 현재가: {current_price:,}{price_unit}
        - 전일 대비: {basic_analysis['price_change_pct']:.2f}%
        - RSI: {basic_analysis['technical_indicators']['rsi']:.1f}
        - 추세: {basic_analysis['trend_analysis']['trend']}
        - 최근 20일 종가: {recent_data['Close'].tolist()}
        - 최근 20일 거래량: {recent_data['Volume'].tolist()}
        
        예측 시 고려사항:
        - 현재 글로벌 경제 상황과 금융시장 동향
        - 해당 {asset_type}에 영향을 미치는 주요 뉴스와 이벤트
        - 기술적 지표와 차트 패턴 분석
        - 시장 심리와 투자자 동향
        - 거시경제 지표 및 정책 변화
        
        다음 형식으로 JSON 응답해주세요:
        {{
            "target_date": "{target_date}",
            "direction": "상승" 또는 "하락",
            "probability": 확률 (50.0-95.0 사이의 숫자),
            "predicted_prices": {{
                "open": 예상시가,
                "close": 예상종가,
                "high": 예상최고가,
                "low": 예상최저가
            }},
            "predicted_volume": 예상거래량,
            "reasoning": "예측 이유 (200자 이내)",
            "detailed_reasoning": "상세한 예측 근거 및 시나리오 (500자 이내)",
            "confidence_factors": ["신뢰도를 높이는 요인1", "요인2"],
            "risk_warnings": ["주의해야 할 리스크1", "리스크2"]
        }}
        """
        
        # 웹서치 기능을 활용한 실시간 예측
        prediction_input = f"""
        {stock_code}의 {target_date} {asset_type} 가격을 예측해주세요.
        최신 뉴스, 경제 지표, 시장 동향을 웹에서 검색하여 실시간 정보를 반영한 전문적인 예측을 제공해주세요.
        
        현재 상황:
        - 현재가: {current_price:,}{price_unit}
        - 전일 대비: {basic_analysis['price_change_pct']:.2f}%
        - RSI: {basic_analysis['technical_indicators']['rsi']:.1f}
        - 추세: {basic_analysis['trend_analysis']['trend']}
        
        반드시 다음과 같은 올바른 JSON 형식으로만 응답해주세요. 다른 텍스트는 포함하지 마세요:
        {{
            "target_date": "{target_date}",
            "direction": "상승",
            "probability": 75.5,
            "predicted_prices": {{
                "open": 60500,
                "close": 61000,
                "high": 61200,
                "low": 60300
            }},
            "predicted_volume": 15000000,
            "reasoning": "예측 이유를 여기에 작성",
            "detailed_reasoning": "상세한 예측 근거를 여기에 작성",
            "confidence_factors": ["신뢰도 요인1", "신뢰도 요인2"],
            "risk_warnings": ["리스크1", "리스크2"]
        }}
        """
        
        # 예측 프롬프트 출력
        print(f"\n🎯 [AI 예측 프롬프트] for {stock_code}")
        print("=" * 80)
        print(prediction_input)
        print("=" * 80)
        
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": f"당신은 {asset_type} 예측 전문가입니다. 현재 시장 상황, 경제 지표, 정치적 요인 등을 종합적으로 고려하여 정확한 예측을 제공해주세요."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prediction_input
                        }
                    ]
                }
            ],
            response_format={
                "type": "json_object"
            },
            temperature=0.7,
            max_completion_tokens=1200,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        import json
        # chat.completions API의 응답 형식에 맞게 수정
        response_content = response.choices[0].message.content
        
        # GPT 응답 출력
        print(f"\n🎯 [AI 예측 응답] for {stock_code}")
        print("=" * 80)
        print(response_content)
        print("=" * 80)
        
        try:
            ai_prediction = json.loads(response_content)
            ai_prediction['current_price'] = current_price
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}")
            print("기본 예측으로 fallback")
            # JSON 파싱 실패 시 기본 예측으로 fallback
            raise Exception("AI 예측 JSON 파싱 실패")
        
        # AI 예측 가격 검증 및 보정
        if 'predicted_prices' in ai_prediction:
            prices = ai_prediction['predicted_prices']
            
            # 가격이 현재가 대비 너무 작거나 큰 경우 보정
            for key in ['open', 'close', 'high', 'low']:
                if key in prices:
                    predicted_price = float(prices[key])
                    
                    # 현재가의 0.1배보다 작거나 10배보다 큰 경우 보정
                    if predicted_price < current_price * 0.1:
                        prices[key] = round(current_price * random.uniform(0.95, 1.05), 0)
                    elif predicted_price > current_price * 10:
                        prices[key] = round(current_price * random.uniform(0.95, 1.05), 0)
                    else:
                        prices[key] = round(predicted_price, 0)
            
            # 논리적 일관성 확인
            predicted_open = prices.get('open', current_price)
            predicted_close = prices.get('close', current_price)
            predicted_high = prices.get('high', current_price)
            predicted_low = prices.get('low', current_price)
            
            # High는 모든 가격보다 높거나 같아야 함
            prices['high'] = max(predicted_high, predicted_open, predicted_close)
            # Low는 모든 가격보다 낮거나 같아야 함
            prices['low'] = min(predicted_low, predicted_open, predicted_close)
        
        return ai_prediction
        
    except Exception as e:
        print(f"AI 예측 오류: {e}")
        # 기본 예측으로 fallback
        return generate_stock_prediction(df)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)