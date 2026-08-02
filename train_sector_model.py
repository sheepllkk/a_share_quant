# train_sector_models.py
import akshare as ak
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# 定义不同板块对应的股票池
SECTORS = {
    "tech_sector_model.pkl": ["sh688981", "sz300750", "sz300124"],        # 科技/半导体/新能源
    "consumption_sector_model.pkl": ["sh600519", "sz000858", "sh600887"], # 消费/白酒/食品
    "finance_sector_model.pkl": ["sh600036", "sh601398", "sh601288"]      # 银行/金融
}

def train_sector_models():
    os.makedirs("model", exist_ok=True)
    
    for model_name, tickers in SECTORS.items():
        print(f"正在训练板块模型: {model_name}，包含股票: {tickers}")
        all_data = []
        
        for ticker in tickers:
            try:
                # 获取历史日线数据
                df = ak.stock_zh_a_daily(symbol=ticker, start_date="20220101", end_date="20260101")
                df['returns'] = df['close'].pct_change()
                df['ema_15'] = df['close'].ewm(span=15, adjust=False).mean()
                df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
                df['ema_diff'] = (df['ema_15'] - df['ema_50']) / df['ema_50']
                df['volatility'] = df['returns'].rolling(5).std()
                
                # 标签：未来一天上涨为1，否则为0
                df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
                df.dropna(inplace=True)
                all_data.append(df[['returns', 'ema_diff', 'volatility', 'target']])
            except Exception as e:
                print(f"获取 {ticker} 数据出错: {e}")
                
        if all_data:
            combined_df = pd.concat(all_data, axis=0)
            X = combined_df[['returns', 'ema_diff', 'volatility']]
            y = combined_df['target']
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            save_path = os.path.join("model", model_name)
            joblib.dump(model, save_path)
            print(f"✅ 板块模型 {model_name} 训练完成并保存！")

if __name__ == "__main__":
    train_sector_models()