# executor.py
import requests
import time
import schedule
from datetime import datetime

# ---------------------------------------------------------
# 🚨 模拟 QMT 券商接口 (xtquant)
# 在实盘中，你需要 pip install xtquant 并正确配置你的资金账号
# 这里用 Dummy 类模拟真实的券商下单动作
# ---------------------------------------------------------
class DummyQMTBroker:
    def __init__(self, account_id):
        self.account_id = account_id
        print(f"✅ [QMT Broker] 券商交易接口初始化成功，账户: {self.account_id}")

    def get_asset(self):
        # 模拟返回当前账户资产
        return {"cash": 150000.0, "total_value": 300000.0}

    def order_target_volume(self, ticker, target_volume):
        print(f"💰 [QMT Broker] 发送实盘指令 -> 代码: {ticker} | 目标持仓调至: {target_volume} 股")
        return True

# ---------------------------------------------------------
# 🧠 核心交易执行逻辑
# ---------------------------------------------------------
def daily_rotation_task():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🚀 开始执行每日截面轮动任务...")
    
    # 1. 初始化券商接口
    broker = DummyQMTBroker(account_id="88888888")
    asset_info = broker.get_asset()
    
    # 2. 向 BASS 系统的 FastAPI 后端请求指令
    api_url = "http://127.0.0.1:8000/get_portfolio_signals"
    payload = {
        "current_cash": asset_info["cash"],
        "current_portfolio_val": asset_info["total_value"],
        "top_k": 3
    }
    
    try:
        print("📡 正在呼叫 BASS 后端获取 AI 打分榜单...")
        response = requests.post(api_url, json=payload, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            
            # 3. 拦截风控报警
            if not data.get("passed_risk_check", False):
                print(f"🚨 触发系统风控！原因: {data.get('reason')} -> 准备执行一键清仓！")
                # broker.order_target_volume(all_stocks, 0)
                return
                
            action = data.get("action")
            
            # 4. 执行交易指令
            if action == "HOLD_CASH":
                print("🟡 AI 综合判断当前无高胜率机会，今日不调仓，持币观望。")
            
            elif action == "ALLOCATE":
                top_picks = data.get("top_picks", [])
                print(f"🟢 接收到建仓指令，今日选股 Top {len(top_picks)}：")
                
                # 简单的等权资金分配计算
                allocated_cash_per_stock = asset_info["total_value"] / len(top_picks)
                
                for pick in top_picks:
                    ticker = pick['ticker']
                    # 假设这里有一个 get_current_price(ticker) 的函数，这里简化处理假定每股 100 元
                    mock_price = 100.0 
                    target_shares = int(allocated_cash_per_stock / mock_price / 100) * 100 # 向下取整到整百股 (一手)
                    
                    print(f"  - 标的: {pick['name']} ({ticker}) | 胜率: {pick['probability']*100:.1f}% | 计划分配: {allocated_cash_per_stock:.2f}元")
                    
                    # 调用券商接口真实下单
                    broker.order_target_volume(ticker, target_shares)
                    
            print("✅ 今日自动化交易执行完毕！")
            
        else:
            print(f"❌ 后端返回异常状态码: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ 无法连接到 BASS 后端，请检查 FastAPI 是否正在运行。错误: {e}")

# ---------------------------------------------------------
# ⏰ 定时任务调度器 (每天下午 14:50 执行，博取尾盘确定性)
# ---------------------------------------------------------
if __name__ == "__main__":
    print("🤖 BASS 无人值守交易机器人已启动，等待交易时间...")
    
    # 在实盘中，我们会设置为每天 14:50 运行
    # schedule.every().day.at("14:50").do(daily_rotation_task)
    
    # 为了测试方便，这里设置为启动时立即运行一次，然后每 10 秒运行一次
    daily_rotation_task()
    schedule.every(10).seconds.do(daily_rotation_task)
    
    while True:
        schedule.run_pending()
        time.sleep(1)