# train_model.py
import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb
import baostock as bs

# 🌟 自动清除代理环境变量，防止 Baostock 登录失败
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['all_proxy'] = ''
os.environ['ALL_PROXY'] = ''

# 🌟 核心资产池 (和 main.py 完全对齐)
UNIVERSE = {
    "sh600519": "贵州茅台", "sz000858": "五粮液",
    "sh600036": "招商银行", "sh601318": "中国平安",
    "sz300750": "宁德时代", "sz002594": "比亚迪",
    "sh688981": "中芯国际", "sz002475": "立讯精密",
    "sz300059": "东方财富", "sh600570": "恒生电子"
}

def build_training_data():
    all_stock_dfs = []

    # 1. 登录 Baostock
    lg = bs.login()
    if lg.error_code != '0':
        raise Exception(f"Baostock 登录失败: {lg.error_msg}")

    print(f"📡 开始拉取 {len(UNIVERSE)} 只核心资产的训练数据...")

    # 2. 先拉取大盘数据 (000001.SH)
    rs_index = bs.query_history_k_data_plus(
        "sh.000001", "date,close",
        start_date="2018-01-01", end_date="2026-12-31",
        frequency="d", adjustflag="3"
    )
    index_data = []
    while (rs_index.error_code == '0') & rs_index.next():
        index_data.append(rs_index.get_row_data())
    
    df_index = pd.DataFrame(index_data, columns=rs_index.fields)
    df_index['date'] = pd.to_datetime(df_index['date'])
    df_index.set_index('date', inplace=True)
    df_index = df_index.astype(float)['close']
    df_index.name = 'index_close'

    # 3. 循环抓取每只股票的数据
    for symbol, name in UNIVERSE.items():
        try:
            bs_symbol = f"sh.{symbol[2:]}" if symbol.startswith("sh") else f"sz.{symbol[2:]}"
            
            # 获取个股量价 + 基本面 (peTTM, pbMRQ)
            rs_stock = bs.query_history_k_data_plus(
                bs_symbol,
                "date,open,close,volume,peTTM,pbMRQ",
                start_date="2018-01-01", end_date="2026-12-31",
                frequency="d", adjustflag="3"
            )
            
            data_list = []
            while (rs_stock.error_code == '0') & rs_stock.next():
                data_list.append(rs_stock.get_row_data())
                
            if not data_list:
                print(f"⚠️ 跳过 {symbol}，无数据")
                continue
                
            df = pd.DataFrame(data_list, columns=rs_stock.fields)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.replace("", np.nan, inplace=True)
            df = df.astype(float)

            # 合并大盘数据
            df = df.join(df_index, how='left')
            df['index_close'] = df['index_close'].ffill().bfill()

            # 🌟 10 维核心特征工程 (加入量能与乖离率)
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
            
            # 🆕 新增特征：量能突变与均线乖离
            df['vol_ratio'] = df['volume'] / df['volume'].rolling(5).mean() # 量比
            df['bias_20'] = df['close'] / df['close'].rolling(20).mean() - 1 # 20日乖离率，捕捉均值回归

            # 🌟 目标函数重构：从"预测绝对收益"改为"预测次日是否跑赢大盘"
            df['next_1d_ret'] = df['close'].shift(-1) / df['close'] - 1
            df['next_1d_idx'] = df['index_close'].shift(-1) / df['index_close'] - 1
            df['target'] = (df['next_1d_ret'] > df['next_1d_idx']).astype(int)

            # 清理缺失值
            df.dropna(inplace=True)
            if df.empty:
                continue
                
            df['ticker'] = symbol
            all_stock_dfs.append(df)
            print(f"✅ 成功加载: {symbol} ({name}), 有效天数: {len(df)}")

        except Exception as e:
            print(f"❌ 加载 {symbol} 失败: {e}")
            continue

    # 登出
    bs.logout()

    if not all_stock_dfs:
        raise ValueError("所有标的训练数据拉取失败。")
        
    # 拼接所有数据
    return pd.concat(all_stock_dfs, axis=0)

def train_and_save():
    print("🚀 开始全市场多标的 8 维特征面板数据构建...")
    
    # 1. 获取拼接好的全部数据
    full_df = build_training_data()
    
    feature_cols = [
        'peTTM', 'pbMRQ',                  
        'relative_strength', 'market_panic', 
        'returns', 'volatility', 'mom_5', 'macd',
        'vol_ratio', 'bias_20'  # 🆕 注册新特征
    ]

    # 2. 定义截面归一化函数
    def cross_section_normalize(df, cols):
        for col in cols:
            # 🌟 修复点：使用 level=0 防止 Date/date 大小写索引报错
            df[col] = df.groupby(level=0)[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8) if len(x) > 1 else 0
            )
        return df

    # 3. 直接对 full_df 执行归一化
    full_df = cross_section_normalize(full_df, feature_cols)
    full_df.dropna(inplace=True)

    # 4. 归一化清洗完毕后，再切分训练集
    train_df = full_df[full_df.index < "2024-01-01"]
    if len(train_df) < 100:
        print("❌ 训练样本不足，请检查日期范围是否包含 2018-2023 年数据。")
        return

    X_train = train_df[feature_cols]
    y_train = train_df['target']
    
    print(f"🧠 开始训练 XGBoost 模型，样本量: {len(X_train)}...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # 🌟 修复点：使用绝对路径，确保无论从终端还是网页端调用，都保存在根目录下的 model 文件夹
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "general_market_model.pkl")
    
    joblib.dump(model, model_path)
    print(f"🎉 炼丹大成功！8 维特征模型已覆盖保存至: {model_path}")

if __name__ == "__main__":
    train_and_save()