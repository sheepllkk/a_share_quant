import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import joblib
import baostock as bs
import xgboost as xgb
import yfinance as yf
import json
import sys
import importlib

# 将项目根目录添加到 Python 模块搜索路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_model

# 导入风控模块 (从原 main.py 迁移过来)
try:
    from core.risk_manager import RiskManager
except ImportError:
    st.error("警告: 未找到 core.risk_manager 模块，请确保文件路径正确。")
    RiskManager = None

st.set_page_config(page_title="A股量化实战系统", layout="wide")
st.title("A股量化多因子策略实战系统 (BASS框架)")

# ==========================================
# 0. 全局配置与模型缓存 (替代原 FastAPI 的启动加载)
# ==========================================
UNIVERSE_MAPPING = {
    "sh600519": "贵州茅台", "sz000858": "五粮液", "sh600036": "招商银行", "sh601318": "中国平安",
    "sz300750": "宁德时代", "sz002594": "比亚迪", "sh688981": "中芯国际", "sz002475": "立讯精密",
    "sz300059": "东方财富", "sh600570": "恒生电子"
}

@st.cache_resource
def load_xgboost_model():
    """缓存加载核心预测模型，避免每次交互重复读取硬盘"""
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "general_market_model.pkl")
    # 兼容相对路径
    if not os.path.exists(model_path):
        model_path = "model/general_market_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# ==========================================
# 1. 定义本地文件缓存辅助功能 (实现刷新持久化)
# ==========================================
CACHE_DIR = "data_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def save_result_to_file(filename, data):
    file_path = os.path.join(CACHE_DIR, filename)
    def convert_to_serializable(obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return obj

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, default=convert_to_serializable, ensure_ascii=False)

def load_result_from_file(filename):
    file_path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            return None
    return None

# ==========================================
# 2. 侧边栏：多页面导航系统
# ==========================================
st.sidebar.header("导航控制台")
page = st.sidebar.radio(
    "请选择操作模式",
    ["策略回测板块", "实盘轮动板块"]
)
st.sidebar.markdown("---")
st.sidebar.subheader("系统与模型管理")

            
st.sidebar.markdown("---")
if page == "策略回测板块":
    st.sidebar.info("在此处进行多因子截面轮动的历史回测评估。")
else:
    st.sidebar.info("在此处实时获取 AI 模型的持仓和交易指令。")

# ==========================================
# 3. 板块一：策略回测 (独立环境)
# ==========================================
if page == "策略回测板块":
    st.subheader("历史回测评估 (OOS)")
    st.caption("基于 2018-2023 数据训练，回测 2024 至今的表现。")
    
    cached_metrics = load_result_from_file("backtest_metrics.json")
    cached_trades = load_result_from_file("backtest_trades.json")
    
    if cached_metrics:
        st.success(f"读取本地缓存回测结果 (生成于 {cached_metrics.get('timestamp', '历史')})")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("组合总收益率", f"{cached_metrics['total_return']*100:.2f}%")
        c2.metric("组合夏普比率", f"{cached_metrics['sharpe_ratio']:.2f}")
        c3.metric("组合最大回撤", f"{cached_metrics['max_drawdown']*100:.2f}%")
        c4.metric("轮动交易胜率", f"{cached_metrics['win_rate']*100:.2f}%")
        
        # 增加容错：防止 equity_curve 为空导致的 KeyError
        if "equity_curve" in cached_metrics and len(cached_metrics["equity_curve"]) > 0:
            st.subheader("策略累计净值走势")
            df_curve = pd.DataFrame(cached_metrics["equity_curve"])
            df_curve['date'] = pd.to_datetime(df_curve['date'])
            df_curve.set_index('date', inplace=True)
            st.line_chart(df_curve[['策略净值', '大盘基准']])
        else:
            st.warning("⚠️ 净值曲线数据为空，可能是历史回测区间内未产生有效交易数据。")

        if cached_trades:
            with st.expander("查看历史每日调仓与持仓明细表", expanded=False):
                df_trades = pd.DataFrame(cached_trades)
                st.dataframe(
                    df_trades,
                    use_container_width=True,
                    column_config={
                        "date": "交易日期",
                        "selected_stocks": "当日选中持仓股票",
                        "top_scores": "预测胜率",
                        "daily_return": st.column_config.NumberColumn("当日组合收益率", format="%.2f%%")
                    }
                )

    if st.sidebar.button("运行组合轮动回测评估 (OOS)", use_container_width=True):
        with st.spinner("正在并发拉取 10 只核心资产历史数据，构建多因子截面轮动回测模型..."):
            try:
                df_index = yf.download("000001.SS", start="2018-01-01", end="2026-12-31", progress=False)
                index_close = df_index['Close'].iloc[:, 0] if isinstance(df_index.columns, pd.MultiIndex) else df_index['Close']
                index_close.name = 'index_close'
                index_close.index = pd.to_datetime(index_close.index).tz_localize(None)
                
                all_stock_dfs = {}
                bs.login()
                
                for symbol, name in UNIVERSE_MAPPING.items():
                    try:
                        code = symbol[2:]
                        suffix = ".SS" if symbol.startswith("sh") else ".SZ"
                        yahoo_ticker = f"{code}{suffix}"
                        
                        df_stock = yf.download(yahoo_ticker, start="2018-01-01", end="2026-12-31", progress=False)
                        if df_stock.empty: continue
                        if isinstance(df_stock.columns, pd.MultiIndex):
                            df_stock.columns = df_stock.columns.get_level_values(0)
                        df_stock.rename(columns={'Open': 'open', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
                        df_stock.index = pd.to_datetime(df_stock.index).tz_localize(None)
                        
                        df = df_stock.join(index_close, how='left')
                        df['index_close'] = df['index_close'].ffill().bfill()
                        
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
                            
                        df['returns'] = df['close'].pct_change()
                        df['volatility'] = df['returns'].rolling(5).std()
                        df['mom_5'] = df['close'].pct_change(5)
                        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
                        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
                        df['macd'] = ema_12 - ema_26
                        
                        df['trade_return'] = df['close'].pct_change()
                        df['index_return'] = df['index_close'].pct_change()
                        df['relative_strength'] = df['trade_return'] - df['index_return']
                        df['market_panic'] = df['index_return'].rolling(20).std()
                        
                        df['vol_ratio'] = df['volume'] / df['volume'].rolling(5).mean()
                        df['bias_20'] = df['close'] / df['close'].rolling(20).mean() - 1
                        
                        df['next_3d_return'] = df['close'].shift(-3) / df['close'] - 1
                        df['target'] = (df['next_3d_return'] > 0.005).astype(int)
                        df['ticker'] = symbol
                        df.dropna(inplace=True)
                        all_stock_dfs[symbol] = df
                    except Exception as ex:
                        print(f"Load {symbol} error: {ex}")
                        continue
                bs.logout()
                
                if not all_stock_dfs:
                    raise ValueError("未能成功加载任何标的历史数据。")

                full_df = pd.concat(all_stock_dfs.values(), axis=0)
                
                # 特征对齐：10 维
                feature_cols = [
                    'peTTM', 'pbMRQ', 'relative_strength', 'market_panic', 
                    'returns', 'volatility', 'mom_5', 'macd',
                    'vol_ratio', 'bias_20'
                ]
                
                for col in feature_cols:
                    full_df[col] = full_df.groupby(level=0)[col].transform(
                        lambda x: (x - x.mean()) / (x.std() + 1e-8) if len(x) > 1 else 0
                    )
                
                train_df = full_df[full_df.index < "2024-01-01"]
                test_df = full_df[full_df.index >= "2024-01-01"]
                
                if test_df.empty:
                    st.error("2024年以后的测试集数据为空，无法生成回测！请检查数据源。")
                    st.stop()
                
                X_train, y_train = train_df[feature_cols], train_df['target']
                model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, min_child_weight=3, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)

                trade_logs = []
                portfolio_daily_returns = []
                benchmark_returns = []
                
                test_dates = sorted(test_df.index.unique())
                index_sma_20 = index_close.rolling(window=20).mean()
                ret_pivot = pd.DataFrame({s: df['trade_return'] for s, df in all_stock_dfs.items()})
                
                for d in test_dates:
                    daily_features_list = []
                    valid_tickers = []
                    for s, df in all_stock_dfs.items():
                        if d in df.index:
                            daily_features_list.append(df.loc[[d], feature_cols])
                            valid_tickers.append(s)
                            
                    if not daily_features_list:
                        portfolio_daily_returns.append(0.0)
                        bench_ret = index_close.pct_change().loc[d] if d in index_close.index else 0.0
                        benchmark_returns.append(0.0 if np.isnan(bench_ret) else float(bench_ret))
                        continue
                        
                    X_today = pd.concat(daily_features_list, axis=0)
                    probs = model.predict_proba(X_today)[:, 1]
                    scored = [{"ticker": vt, "prob": p} for vt, p in zip(valid_tickers, probs)]
                    scored.sort(key=lambda x: x['prob'], reverse=True)
                    top_picks = [item for item in scored[:3] if item['prob'] > 0.50]
                    
                    if top_picks and d in ret_pivot.index:
                        tickers = [p['ticker'] for p in top_picks]
                        names = [UNIVERSE_MAPPING.get(t, t) for t in tickers]
                        probs_str = [f"{p['prob']*100:.1f}%" for p in top_picks]
                        
                        day_ret = ret_pivot.loc[d, tickers].mean()
                        if np.isnan(day_ret): day_ret = 0.0
                        
                        trade_logs.append({
                            "date": d.strftime("%Y-%m-%d"),
                            "selected_stocks": ", ".join(names),
                            "top_scores": ", ".join(probs_str),
                            "daily_return": float(day_ret * 100)
                        })
                    else:
                        day_ret = 0.0
                        trade_logs.append({
                            "date": d.strftime("%Y-%m-%d"),
                            "selected_stocks": "空仓观望 (未达到买入阈值)",
                            "top_scores": "-",
                            "daily_return": 0.0
                        })
                        
                    portfolio_daily_returns.append(day_ret)
                    
                    bench_ret = index_close.pct_change().loc[d] if d in index_close.index else 0.0
                    benchmark_returns.append(0.0 if np.isnan(bench_ret) else float(bench_ret))

                perf_series = pd.Series(portfolio_daily_returns, index=test_dates)
                bench_series = pd.Series(benchmark_returns, index=test_dates)
                
                cum_returns = (1 + perf_series).cumprod()
                bench_cum = (1 + bench_series).cumprod()
                
                sharpe_ratio = float(np.sqrt(252) * perf_series.mean() / (perf_series.std() + 1e-9))
                rolling_max = cum_returns.cummax()
                drawdown = (cum_returns - rolling_max) / rolling_max
                max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0.0
                win_rate = float((perf_series[perf_series != 0] > 0).sum() / (perf_series != 0).sum()) if (perf_series != 0).sum() > 0 else 0.0

                equity_curve = []
                for dt, strat_v, bench_v in zip(test_dates, cum_returns, bench_cum):
                    equity_curve.append({
                        "date": dt.strftime("%Y-%m-%d"),
                        "策略净值": float(strat_v),
                        "大盘基准": float(bench_v)
                    })

                metrics = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_return": float(cum_returns.iloc[-1] - 1) if len(cum_returns) > 0 else 0.0,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown,
                    "win_rate": win_rate,
                    "equity_curve": equity_curve 
                }
                save_result_to_file("backtest_metrics.json", metrics)
                save_result_to_file("backtest_trades.json", trade_logs)
                
                st.rerun()

            except Exception as e:
                import traceback
                st.error(f"组合回测计算异常: {str(e)}")
                st.code(traceback.format_exc())

# ==========================================
# 4. 板块二：实盘轮动指令 (完全集成后端逻辑)
# ==========================================
def run_realtime_scan(current_cash, current_portfolio_val, top_k):
    """原生集成原 main.py 的逻辑，不再依赖 HTTP 端口请求"""
    model = load_xgboost_model()
    if model is None:
        raise Exception("未找到通用的量化预测模型文件，请先在左侧菜单点击“重新训练核心模型”！")
        
    # 风控检查
    if RiskManager is not None:
        risk_mgr = RiskManager(max_drawdown_pct=0.15)
        is_safe = risk_mgr.check_portfolio_risk(current_portfolio_val)
        if not is_safe:
            return {
                "action": "FORCE_LIQUIDATE",
                "reason": "触发总资产风控熔断，执行全部强平",
                "passed_risk_check": False
            }
    
    # 登录 Baostock
    lg = bs.login()
    if lg.error_code != '0':
        raise Exception(f"Baostock 登录失败: {lg.error_msg}")
        
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
        raise Exception(f"大盘数据获取失败: {str(e)}")

    scored_stocks = []

    for symbol, name in UNIVERSE_MAPPING.items():
        try:
            bs_symbol = f"sh.{symbol[2:]}" if symbol.startswith("sh") else f"sz.{symbol[2:]}"
            
            rs_stock = bs.query_history_k_data_plus(
                bs_symbol, "date,open,close,volume",  
                start_date="2023-12-01", end_date="2026-12-31",
                frequency="d", adjustflag="3"
            )
            
            stock_data_list = []
            while (rs_stock.error_code == '0') & rs_stock.next():
                stock_data_list.append(rs_stock.get_row_data())
                
            if not stock_data_list: continue

            df_stock = pd.DataFrame(stock_data_list, columns=rs_stock.fields)
            df_stock['date'] = pd.to_datetime(df_stock['date'])
            df_stock.set_index('date', inplace=True)
            df_stock.replace("", np.nan, inplace=True)
            df_stock = df_stock.astype(float)

            df = df_stock.join(index_close, how='left') 
            df['index_close'] = df['index_close'].ffill().bfill()
            
            rs = bs.query_history_k_data_plus(
                bs_symbol, "date,peTTM,pbMRQ", 
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
            
            df.dropna(inplace=True)
            if df.empty: continue
                
            # 10 维特征提取
            feature_cols = [
                'peTTM', 'pbMRQ',                  
                'relative_strength', 'market_panic', 
                'returns', 'volatility', 'mom_5', 'macd',
                'vol_ratio', 'bias_20'
            ]
            latest_features = df[feature_cols].iloc[[-1]]
            pred_prob = float(model.predict_proba(latest_features)[0][1])
            
            current_close = float(df['close'].iloc[-1])
            ma_20 = float(df['close'].rolling(20).mean().iloc[-1]) if len(df) >= 20 else current_close
            
            scored_stocks.append({
                "ticker": symbol,
                "name": name,
                "probability": pred_prob,
                "current_price": current_close,
                "suggested_buy_range": f"¥{current_close*0.995:.2f} - ¥{current_close*1.005:.2f}",
                "stop_loss_price": round(max(ma_20 * 0.95, current_close * 0.93), 2), 
                "target_price": round(current_close * 1.05, 2) 
            })
            
        except Exception as e:
            continue
            
    bs.logout()

    scored_stocks.sort(key=lambda x: x['probability'], reverse=True)
    top_picks = [s for s in scored_stocks[:top_k] if s['probability'] > 0.50]
    
    return {
        "action": "ALLOCATE" if top_picks else "HOLD_CASH",
        "passed_risk_check": True,
        "universe_size": len(UNIVERSE_MAPPING),
        "top_picks": top_picks,
        "all_scores": scored_stocks 
    }


if page == "实盘轮动板块":
    st.subheader("实盘轮动监控指令")
    st.caption("系统将自动扫描全市场核心资产，并执行 AI 智能打分与风控。")
    
    # === 新增：实盘操作标准指南 (SOP) ===
    with st.expander("📖 必读：BASS 核心资产轮动系统 - 实盘操作指南 (SOP)", expanded=True):
        st.markdown("""
        **本系统基于 AI 深度学习，执行【日频动态截面轮动】策略。为保证实盘收益与回测一致，请务必严格遵守以下操作纪律：**

        *   ⏰ **扫描时间（关键）**：请在**每个交易日的 14:45 - 14:50（尾盘）**点击扫描获取最新指令，避免盘中价格剧烈波动。
        *   🟢 **买入建仓**：若您当前空仓或有可用资金，请参考 AI 给出的 Top 推荐名单，并在 **[建议买入区间]** 内进行等权建仓。
        *   🟡 **持仓与调仓（如何卖出）**：
            *   **若已持仓标的【继续出现】在今日推荐名单中**：代表 AI 判定其明日仍有较高胜率，**继续持有**。
            *   **若已持仓标的【消失】在今日推荐名单中**：代表其上涨动能衰竭或被其他股票挤出，**请于尾盘果断卖出**，并将资金切换至今日新上榜的标的。
        *   🔴 **铁血风控（止损/止盈）**：
            *   **止损**：盘中任何时候，一旦有效跌破给定的 **[动态止损参考]** 价位，请无条件止损离场，保住本金！
            *   **止盈**：若触及 **[短线目标位]**，您可以根据自身风险偏好选择落袋为安，或继续跟随 AI 信号持有。
        
        > 💡 **核心逻辑**：AI 每天都会在核心资产池中“优中选优”。不在榜单不代表它一定会跌，而是说明有“性价比更高、胜率更大”的选择。请摒弃主观执念，坚决执行轮动指令。
        """)
    # =================================
    
    cached_signals = load_result_from_file("latest_signals.json")
    
    with st.sidebar:
        current_cash = st.number_input("当前可用现金 (¥)", value=150000.0, step=10000.0)
        portfolio_val = st.number_input("当前账户总资产 (¥)", value=300000.0, step=10000.0)
        top_k_input = st.slider("最大持仓数量 (Top K)", min_value=1, max_value=5, value=3)
        check_btn = st.button("扫描全宇宙获取轮动指令", use_container_width=True)

    if check_btn:
        try:
            with st.spinner(f"正在抓取底层资产数据，执行 XGBoost 截面横向打分..."):
                # 直接调用合并到本地的方法，不再走 HTTP requests
                data = run_realtime_scan(current_cash, portfolio_val, top_k_input)
                save_result_to_file("latest_signals.json", data)
                st.rerun() 
        except Exception as e:
            st.error(f"信号生成失败：{str(e)}")

    if cached_signals:
        data = cached_signals
        st.subheader("系统风控与大盘状态")
        col1, col2 = st.columns(2)
        col1.metric("资产池总扫描数", f"{data.get('universe_size', 0)} 只")
        
        if not data.get('passed_risk_check', True):
            col2.metric("风控状态", "触发熔断")
            st.error(f"**强平警告**：{data.get('reason', '资产回撤超限，建议清仓！')}")
        else:
            col2.metric("风控状态", "安全无风险")
            st.subheader("截面轮动交易指令")
            if data.get('action') == "HOLD_CASH":
                st.warning("**当前大盘环境恶劣或无高胜率标的，AI 建议：持币观望，不予建仓。**")
            elif data.get('action') == "ALLOCATE":
                st.success(f"**AI 建议：执行资金等权分配，买入以下 Top {len(data.get('top_picks', []))} 标的**")
                cols = st.columns(len(data['top_picks']))
                for i, pick in enumerate(data['top_picks']):
                    with cols[i]:
                        st.info(f"**{pick['name']}** ({pick['ticker']})")
                        st.metric("上涨置信度", f"{pick['probability']*100:.1f}%")
                        
                        st.markdown("---")
                        st.markdown(f"* **最新参考价**: ¥{pick.get('current_price', 0):.2f}")
                        st.markdown(f"* **建议买入区间**: {pick.get('suggested_buy_range', '市价附近')}")
                        st.markdown(f"* **动态止损参考**: ¥{pick.get('stop_loss_price', 0):.2f}")
                        st.markdown(f"* **短线目标位**: ¥{pick.get('target_price', 0):.2f}")
            
            with st.expander("查看全景股票池打分排名 (AI Score)"):
                if 'all_scores' in data and data['all_scores']:
                    df_scores = pd.DataFrame(data['all_scores'])
                    df_scores.index += 1
                    df_scores.rename(columns={'ticker':'代码', 'name':'名称', 'probability':'预测胜率'}, inplace=True)
                    if not df_scores.empty and df_scores['预测胜率'].dtype != 'O':
                        df_scores['预测胜率'] = df_scores['预测胜率'].apply(lambda x: f"{x*100:.2f}%")
                    st.dataframe(df_scores, use_container_width=True)
                else:
                    st.info("当前没有获取到有效的截面打分数据。")
    else:
        st.info("💡 当前暂无轮动指令缓存。请在左侧边栏点击“扫描全宇宙获取轮动指令”生成实时信号。")