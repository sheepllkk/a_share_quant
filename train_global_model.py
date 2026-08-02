# train_global_model.py
import akshare as ak
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_and_save_global_model():
    # 选取不同板块的代表性股票作为训练样本池
    tickers = ["sh600519", "sz300750", "sh600036", "sz000001"] 
    all_data = []

    print("正在聚合多股票历史数据用于训练通用模型...")
    for ticker in tickers:
        try:
            df = ak.stock_zh_a_daily(symbol=ticker, start_date="20220101", end_date="20260101")
            df['returns'] = df['close'].pct_change()
            df['ema_15'] = df['close'].ewm(span=15, adjust=False).mean()
            df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['ema_diff'] = (df['ema_15'] - df['ema_50']) / df['ema_50']
            df['volatility'] = df['returns'].rolling(5).std()
            
            # 标签：未来一天收益率大于0为1，否则为0
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            df.dropna(inplace=True)
            all_data.append(df[['returns', 'ema_diff', 'volatility', 'target']])
        except Exception as e:
            print(f"获取 {ticker} 数据失败: {e}")

    # 合并所有股票的数据集
    combined_df = pd.concat(all_data, axis=0)
    X = combined_df[['returns', 'ema_diff', 'volatility']]
    y = combined_df['target']

    # 训练随机森林大模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/general_market_model.pkl")
    print("通用大盘混合模型已成功训练并保存至 model/general_market_model.pkl！")

if __name__ == "__main__":
    train_and_save_global_model()