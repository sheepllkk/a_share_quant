# 修改 core/risk_manager.py 的头部
import os
import logging

# 自动检查并创建 logs 文件夹
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename='logs/trading_system.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class RiskManager:
    def __init__(self, max_drawdown_pct=0.15, single_loss_limit=0.05):
        self.max_drawdown_pct = max_drawdown_pct # 最大回撤阈值 (如 15%)
        self.single_loss_limit = single_loss_limit # 单笔最大亏损限制
        self.peak_value = 0.0

    def check_portfolio_risk(self, current_portfolio_value: float) -> bool:
        """
        检查整体资产组合是否触发熔断机制
        """
        if current_portfolio_value > self.peak_value:
            self.peak_value = current_portfolio_value
            
        if self.peak_value > 0:
            current_drawdown = (self.peak_value - current_portfolio_value) / self.peak_value
            if current_drawdown >= self.max_drawdown_pct:
                msg = f"【风险警报】触发组合级熔断！当前回撤 {current_drawdown*100:.2f}% 超过阈值 {self.max_drawdown_pct*100}%"
                logging.critical(msg)
                print(msg)
                return False # 返回 False 代表禁止继续交易，需强制清仓
                
        return True