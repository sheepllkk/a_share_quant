# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb
import baostock as bs
from core.risk_manager import RiskManager

# 清除代理环境变量干扰
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['all_proxy'] = ''
os.environ['ALL_PROXY'] = ''

app = FastAPI(title="A-Share Portfolio Rotation API", version="4.2.0")

# 加载全局 XGBoost 模型 (10维特征版)
model_path = "model/general_market_model.pkl"
MODELS = {
    "general": joblib.load(model_path) if os.path.exists(model_path) else None
}

# 核心资产固定池
UNIVERSE = {
    "sh600519": "贵州茅台", "sz000858": "五粮液",
    "sh600036": "招商银行", "sh601318": "中国平安",
    "sz300750": "宁德时代", "sz002594": "比亚迪",
    "sh688981": "中芯国际", "sz002475": "立讯精密",
    "sz300059": "东方财富", "sh600570": "恒生电子"
}

class PortfolioRequest(BaseModel):
    current_cash: float
    current_portfolio_val: float
    top_k: int = 3

risk_mgr = RiskManager(max_drawdown_pct=0.15)

@app.post("/get_portfolio_signals")
def generate_portfolio_signals(req: PortfolioRequest):
    model = MODELS.get("general")
    if model is None:
        raise HTTPException(status_code=500, detail="未找到通用的量化预测模型文件，请先在网页端点击重新训练模型！")
        
    is_safe = risk_mgr.check_portfolio_risk(req.current_portfolio_val)
    if not is_safe:
        return {
            "action": "FORCE_LIQUIDATE",
            "reason": "触发总资产风控熔断，执行全部强平",
            "passed_risk_check": False
        }
    
    # 登录 Baostock
    lg = bs.login()
    if lg.error_code != '0':
        raise HTTPException(status_code=500, detail=f"Baostock 登录失败: {lg.error_msg}")
        
    try:
        rs_index = bs.query_history_k_data_plus(
            "sh.000001", "date,close", 
            start_date="2023-12-01", end_date="2026-12-31",
            frequency="d", adjustflag="3"
        )
        index_data_list = []
        while (rs_index.error_code == '0') & rs_index.next():
            index_data_list.append(rs_index.get_row_data())
            
        df_index = pd.DataFrame(index_data_list, columns=rs_index.fields)
        df_index['date'] = pd.to_datetime(df_index['date'])
        df_index.set_index('date', inplace=True)
        df_index = df_index.astype(float)
        index_close = df_index['close']
        index_close.name = 'index_close'
    except Exception as e:
        bs.logout()
        raise HTTPException(status_code=500, detail=f"大盘数据获取失败: {str(e)}")

    scored_stocks = []

    for symbol, name in UNIVERSE.items():
        try:
            bs_symbol = f"sh.{symbol[2:]}" if symbol.startswith("sh") else f"sz.{symbol[2:]}"
            
            # 1. 获取个股量价数据
            rs_stock = bs.query_history_k_data_plus(
                bs_symbol,
                "date,open,close,volume",  
                start_date="2023-12-01", end_date="2026-12-31",
                frequency="d", adjustflag="3"
            )
            
            stock_data_list = []
            while (rs_stock.error_code == '0') & rs_stock.next():
                stock_data_list.append(rs_stock.get_row_data())
                
            if not stock_data_list:
                continue

            df_stock = pd.DataFrame(stock_data_list, columns=rs_stock.fields)
            df_stock['date'] = pd.to_datetime(df_stock['date'])
            df_stock.set_index('date', inplace=True)
            df_stock.replace("", np.nan, inplace=True)
            df_stock = df_stock.astype(float)

            # 2. 合并大盘数据
            df = df_stock.join(index_close, how='left') 
            df['index_close'] = df['index_close'].ffill().bfill()
            
            # 3. 获取基本面数据 (Baostock)
            rs = bs.query_history_k_data_plus(
                bs_symbol,
                "date,peTTM,pbMRQ", 
                start_date="2024-01-01", end_date="2026-12-31",
                frequency="d", adjustflag="3"
            )
            
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
                
            if data_list:
                df_fundamental = pd.DataFrame(data_list, columns=rs.fields)
                df_fundamental['date'] = pd.to_datetime(df_fundamental['date'])
                df_fundamental.set_index('date', inplace=True)
                df_fundamental.replace("", np.nan, inplace=True)
                df_fundamental = df_fundamental.astype(float).ffill()
                df = df.join(df_fundamental, how='left')
                df['peTTM'] = df['peTTM'].ffill()
                df['pbMRQ'] = df['pbMRQ'].ffill()
            else:
                df['peTTM'] = np.nan
                df['pbMRQ'] = np.nan

            # 4. 完整的 10 维特征工程 (严格与 train_model.py 对齐)
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(5).std()
            df['mom_5'] = df['close'].pct_change(5)
            
            ema_12 = df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema_12 - ema_26
            
            df['trade_return'] = df['close'].pct_change() 
            df['index_return'] = df['index_close'].pct_change(fill_method=None)
            df['relative_strength'] = df['trade_return'] - df['index_return']
            df['market_panic'] = df['index_return'].rolling(20).std() 
            
            # 🆕 补齐漏掉的两个特征列
            df['vol_ratio'] = df['volume'] / df['volume'].rolling(5).mean()
            df['bias_20'] = df['close'] / df['close'].rolling(20).mean() - 1
            
            df.dropna(inplace=True)
            if df.empty:
                continue
                
            # 10 维特征严格对齐
            feature_cols = [
                'peTTM', 'pbMRQ',                  
                'relative_strength', 'market_panic', 
                'returns', 'volatility', 'mom_5', 'macd',
                'vol_ratio', 'bias_20'
            ]
            latest_features = df[feature_cols].iloc[[-1]]
            pred_prob = float(model.predict_proba(latest_features)[0][1])
            
            scored_stocks.append({
                "ticker": symbol,
                "name": name,
                "probability": pred_prob
            })
            
        except Exception as e:
            print(f"❌ 跳过 {symbol}，原因：{str(e)}")
            continue
            
    bs.logout()

    scored_stocks.sort(key=lambda x: x['probability'], reverse=True)
    top_picks = [s for s in scored_stocks[:req.top_k] if s['probability'] > 0.50]
    
    return {
        "action": "ALLOCATE" if top_picks else "HOLD_CASH",
        "passed_risk_check": True,
        "universe_size": len(UNIVERSE),
        "top_picks": top_picks,
        "all_scores": scored_stocks 
    }