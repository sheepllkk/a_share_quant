# train_model.py
import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb
import yfinance as yf
import baostock as bs  # 🆕 新增：用于获取基本面特征

# 🌟 核心资产池
UNIVERSE_MAPPING = {
    "sh600519": "贵州茅台", "sz000858": "五粮液",
    "sh600036": "招商银行", "sh601318": "中国平安",
    "sz300750": "宁德时代", "sz002594": "比亚迪",
    "sh688981": "中芯国际", "sz002475": "立讯精密",
    "sz300059": "东方财富", "sh600570": "恒生电子"
}

def build_training_data():
    all_stock_dfs = []

    print(f"📡 开始拉取 {len(UNIVERSE_MAPPING)} 只核心资产的训练数据 (包含基本面数据)...")

    # 1. 获取大盘数据 (上证指数 000001.SS)
    df_index = yf.download("000001.SS", start="2018-01-01", end="2026-12-31", progress=False)
    if isinstance(df_index.columns, pd.MultiIndex):
        df_index.columns = df_index.columns.get_level_values(0)
    index_close = df_index['Close']
    index_close.name = 'index_close'
    index_close.index = pd.to_datetime(index_close.index).tz_localize(None)
    
    # 🆕 登录 Baostock
    bs.login()

    # 2. 循环抓取每只股票的数据
    for symbol, name in UNIVERSE_MAPPING.items():
        try:
            # --- yfinance 获取量价数据 ---
            code = symbol[2:]
            suffix = ".SS" if symbol.startswith("sh") else ".SZ"
            yahoo_ticker = f"{code}{suffix}"
            
            df = yf.download(yahoo_ticker, start="2018-01-01", end="2026-12-31", progress=False)
            if df.empty:
                continue
                
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df.rename(columns={'Open': 'open', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.astype(float)

            # 合并大盘数据
            df = df.join(index_close, how='left')
            df['index_close'] = df['index_close'].ffill().bfill()
            
            # 🆕 --- Baostock 获取基本面特征 (peTTM, pbMRQ) ---
            bs_symbol = f"sh.{code}" if symbol.startswith("sh") else f"sz.{code}"
            rs = bs.query_history_k_data_plus(
                bs_symbol, "date,peTTM,pbMRQ",
                start_date="2018-01-01", end_date="2026-12-31",
                frequency="d", adjustflag="3"
            )
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            if data_list:
                df_fund = pd.DataFrame(data_list, columns=rs.fields)
                df_fund['date'] = pd.to_datetime(df_fund['date'])
                df_fund.set_index('date', inplace=True)
                df_fund.replace("", np.nan, inplace=True)
                df_fund = df_fund.astype(float).ffill()
                
                df = df.join(df_fund, how='left')
                df['peTTM'] = df['peTTM'].ffill()
                df['pbMRQ'] = df['pbMRQ'].ffill()
            else:
                df['peTTM'] = np.nan
                df['pbMRQ'] = np.nan

            # 🌟 10 维核心特征工程 (加入 PE/PB)
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
            
            df['vol_ratio'] = df['volume'] / df['volume'].rolling(5).mean() 
            df['bias_20'] = df['close'] / df['close'].rolling(20).mean() - 1 

            # 🌟 目标函数：预测次日是否跑赢大盘
            df['next_1d_ret'] = df['close'].shift(-1) / df['close'] - 1
            df['next_1d_idx'] = df['index_close'].shift(-1) / df['index_close'] - 1
            df['target'] = (df['next_1d_ret'] > df['next_1d_idx']).astype(int)

            df.dropna(inplace=True)
            if df.empty:
                continue
                
            df['ticker'] = symbol
            all_stock_dfs.append(df)
            print(f"✅ 成功加载: {symbol} ({name}), 有效天数: {len(df)}")

        except Exception as e:
            print(f"❌ 加载 {symbol} 失败: {e}")
            continue
            
    # 🆕 登出 Baostock
    bs.logout()

    if not all_stock_dfs:
        raise ValueError("所有标的训练数据拉取失败。")
        
    return pd.concat(all_stock_dfs, axis=0)

def train_and_save():
    print("🚀 开始构建 10 维量价与基本面特征模型...")
    full_df = build_training_data()
    
    # 🆕 特征列表：加入 peTTM 和 pbMRQ 构成 10 维
    feature_cols = [
        'peTTM', 'pbMRQ',
        'relative_strength', 'market_panic', 
        'returns', 'volatility', 'mom_5', 'macd',
        'vol_ratio', 'bias_20'
    ]

    def cross_section_normalize(df, cols):
        for col in cols:
            df[col] = df.groupby(level=0)[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8) if len(x) > 1 else 0
            )
        return df

    full_df = cross_section_normalize(full_df, feature_cols)
    full_df.dropna(inplace=True)

    train_df = full_df[full_df.index < "2024-01-01"]

    X_train = train_df[feature_cols]
    y_train = train_df['target']
    
    print(f"🧠 开始训练 XGBoost 模型，样本量: {len(X_train)}...")
    model = xgb.XGBClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=3, 
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8, 
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "general_market_model.pkl")
    
    joblib.dump(model, model_path)
    print(f"🎉 炼丹大成功！10 维多因子模型已保存至: {model_path}")

if __name__ == "__main__":
    train_and_save()