import pandas as pd
import numpy as np
import akshare as ak
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import os

class QuantEngine:
    def __init__(self, model_path="model/random_forest_quant.pkl", initial_cash=100000.0):
        self.initial_cash = initial_cash
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        """加载本地训练好的机器学习模型 (.pkl)"""
        if os.path.exists(self.model_path):
            return joblib.load(self.model_path)
        return None

    def fetch_and_preprocess_data(self, ticker: str, start_date: str, end_date: str):
        """拉取 A 股数据并进行标准特征工程"""
        df = ak.stock_zh_a_daily(symbol=ticker, start_date=start_date, end_date=end_date)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # 特征工程（与训练时保持一致）
        df['returns'] = df['close'].pct_change()
        df['ema_15'] = df['close'].ewm(span=15, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_diff'] = (df['ema_15'] - df['ema_50']) / df['ema_50']
        df['volatility'] = df['returns'].rolling(5).std()
        df.dropna(inplace=True)
        return df

    def run_backtest(self, ticker: str, start_date: str, end_date: str):
        """
        核心历史回测引擎：模拟过去一段时间的交易表现，并计算 BASS 框架指标
        """
        if self.model is None:
            raise ValueError("未检测到训练好的 ML 模型，请先训练并保存 .pkl 文件。")
            
        df = self.fetch_and_preprocess_data(ticker, start_date, end_date)
        feature_cols = ['returns', 'ema_diff', 'volatility']
        X = df[feature_cols]
        
        # 模型推理预测
        df['signal'] = self.model.predict(X)
        
        # 回测仿真计算（考虑 A 股多头单向交易与基础滑点/手续费）
        cash = self.initial_cash
        holdings = 0
        portfolio_values = []
        buy_cost = 0.0003
        sell_cost = 0.0013 # 含印花税
        
        for i in range(len(df)):
            price = df['close'].iloc[i]
            signal = df['signal'].iloc[i]
            
            # 策略规则：信号为1买入，信号为0卖出
            if signal == 1 and holdings == 0:
                shares = int(cash / (price * (1 + buy_cost)) / 100) * 100
                if shares > 0:
                    holdings = shares
                    cash -= holdings * price * (1 + buy_cost)
            elif signal == 0 and holdings > 0:
                cash += holdings * price * (1 - sell_cost)
                holdings = 0
                
            portfolio_values.append(cash + holdings * price)
            
        df['portfolio_value'] = portfolio_values
        
        # 计算核心绩效指标 (BASS 框架与风控指标)
        returns = df['portfolio_value'].pct_change().dropna()
        total_return = (df['portfolio_value'].iloc[-1] - self.initial_cash) / self.initial_cash
        annual_return = (1 + total_return) ** (252 / len(df)) - 1
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # 最大回撤计算
        cum_max = df['portfolio_value'].cummax()
        drawdown = (df['portfolio_value'] - cum_max) / cum_max
        max_drawdown = drawdown.min()
        
        # 整理输出结果
        history_records = []
        for d, row in df.iterrows():
            history_records.append({
                "date": d.strftime("%Y-%m-%d"),
                "close": row['close'],
                "portfolio_value": row['portfolio_value']
            })
            
        return {
            "metrics": {
                "total_return_pct": round(total_return * 100, 2),
                "annual_return_pct": round(annual_return * 100, 2),
                "sharpe_ratio": round(sharpe_ratio, 3),
                "max_drawdown_pct": round(max_drawdown * 100, 2)
            },
            "history": history_records
        }