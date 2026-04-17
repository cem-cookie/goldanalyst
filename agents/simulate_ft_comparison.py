"""
simulate_ft_comparison.py
对比：
1. GPT-4o-mini（基础模型）
2. GPT-4o-mini（Fine-tuned 模型）

要求：
- 使用你的 GPT4oAgent（已贴出代码）
- 调用 decide() → execute()
- 读取 data/news/{date}/market_analysis.txt
- 读取 data/gold_history.csv
- 输出回测结果和图表
"""

import os
import glob
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from agents.models.gpt4o_agent import GPT4oAgent
from agents.trading_agent import TradingAgent  # ensure imported

# ================================
# CONFIG
# ================================
DATA_ROOT = "data/news"
PRICE_CSV = "data/gold_history.csv"
RESULT_DIR = "results"
PLOT_DIR = os.path.join(RESULT_DIR, "plots")

FT_MODEL = "ft:gpt-4o-mini-2024-07-18:personal:my-experiment-1:CbRSGFvv"


# ==========================================================
# UTILS
# ==========================================================
def ensure_dirs():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)


def load_prices(csv_path):
    """
    Robust CSV loader:
    - 支持有表头 / 无表头
    - 自动找到日期列和价格列
    - 标准化为列名: date, price
    """
    # 1. 先读原文件的第一行看看是不是表头
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]

    first_line = lines[0]
    first_col = first_line.split(",")[0].lower()

    # 如果第一列是 date / Date / DAY 等，就认为有表头
    has_header = first_col in ["date", "day", "time"]

    if has_header:
        df = pd.read_csv(csv_path)
    else:
        # 没有表头：直接当成两列
        df = pd.read_csv(csv_path, header=None, names=["date", "price"])

    # ---- 统一 date 列名 ----
    if "date" not in df.columns:
        # 找一个名字里带 date 的列
        date_col = None
        for c in df.columns:
            if "date" in c.lower():
                date_col = c
                break
        if date_col is None:
            # 最坏情况：就当第一列是日期
            date_col = df.columns[0]
        df = df.rename(columns={date_col: "date"})

    # ---- 统一 price 列名 ----
    if "price" not in df.columns:
        price_col = None
        # 优先找典型的价格列名
        for c in df.columns:
            cl = c.lower()
            if cl in ["price", "close", "gold", "xauusd", "xau", "value"]:
                price_col = c
                break
        # 如果还没找到，就把“非 date 的第二列”当成价格列
        if price_col is None:
            non_date_cols = [c for c in df.columns if c != "date"]
            if len(non_date_cols) == 0:
                raise ValueError(f"[load_prices] Cannot find price column in CSV. Columns = {df.columns.tolist()}")
            price_col = non_date_cols[0]

        df = df.rename(columns={price_col: "price"})

    # ---- 清洗数据 ----
    df = df[["date", "price"]].dropna()

    # 只保留形如 YYYY-MM-DD 的行
    df = df[df["date"].astype(str).str.match(r"\d{4}-\d{2}-\d{2}")]

    # 解析日期
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df = df.dropna(subset=["date"])

    # 解析价格
    df["price"] = df["price"].astype(float)

    print(f"[DEBUG] Loaded prices with columns: {df.columns.tolist()}, rows: {len(df)}")
    return df


def plot_agent_comparison(agents, price_df):
    """对比两个 agent 的净值曲线 + 金价"""
    fig, ax1 = plt.subplots(figsize=(13, 7))

    # 金价曲线（右轴）
    ax2 = ax1.twinx()
    ax2.plot(price_df["date"], price_df["price"], color="#DAA520", linewidth=2.5, label="Gold Price")

    # Agent 净值（左轴）
    colors = ["#3498db", "#e74c3c"]
    for agent, color in zip(agents, colors):
        log = pd.DataFrame(agent.logs)
        log["date"] = pd.to_datetime(log["date"])
        ax1.plot(log["date"], log["equity"],
                 color=color, linewidth=2.2, label=f"{agent.name}")

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Portfolio Equity")
    ax2.set_ylabel("Gold Price")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.grid(True, alpha=0.25)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, "comparison_ft.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ Saved comparison plot → {out_path}")


# ==========================================================
# MAIN BACKTEST
# ==========================================================
def run():
    ensure_dirs()

    print("\n[1] Loading gold prices...")
    price_df = load_prices(PRICE_CSV)
    print(f"Loaded {len(price_df)} price records")

    # -------------------------------------------
    # 加载有 analysis 的日期
    # -------------------------------------------
    news_days = sorted([
        os.path.basename(p) for p in glob.glob(f"{DATA_ROOT}/*")
        if os.path.isdir(p)
    ])

    print("\n[2] Loading news days...")
    print(f"Found {len(news_days)} days")

    # 过滤 price_df 只保留有分析的日期
    price_df["date_str"] = price_df["date"].dt.strftime("%Y-%m-%d")
    price_df = price_df[price_df["date_str"].isin(news_days)].sort_values("date")

    if price_df.empty:
        print("❌ No matching days between price and news folders!")
        return

    print(f"Matched {len(price_df)} trading days.")

    # ------------------------------------------------
    # 初始化两个模型（base vs finetuned）
    # ------------------------------------------------
    print("\n[3] Initializing agents...")

    base_agent = GPT4oAgent(
        name="GPT-4o-mini-base",
        api_key=os.getenv("OPENAI_API_KEY"),
        initial_cash=100_000,
        max_alloc=0.5,
        fee_bps=10
    )

    ft_agent = GPT4oAgent(
        name="GPT-4o-mini-FT",
        api_key=os.getenv("OPENAI_API_KEY"),
        initial_cash=100_000,
        max_alloc=0.5,
        fee_bps=10
    )

    # 修改 FT agent 的模型名
    ft_agent.model = FT_MODEL

    agents = [base_agent, ft_agent]
    print("Initialized 2 agents.")

    # ------------------------------------------------
    # 回测循环
    # ------------------------------------------------
    print("\n[4] Running backtest...")
    print("=" * 60)

    for i, row in price_df.iterrows():
        date_str = row["date_str"]
        price = row["price"]

        # 加载 market_analysis.txt
        path = f"{DATA_ROOT}/{date_str}/market_analysis.txt"
        if not os.path.exists(path):
            summary = "No analysis available."
        else:
            summary = open(path, "r", encoding="utf-8").read()

        print(f"\n[{date_str}] Price = ${price:.2f}")

        # 逐个 agent 做决策
        for agent in agents:
            # 使用决定逻辑
            decision = agent.decide(summary, price)
            agent.execute(decision, price, date_str)

            last = agent.logs[-1]
            print(f"  {agent.name:<18} {last['action']:>5} {last['amount_oz']:>6.2f} oz "
                  f"{'✓' if last['executed'] else '✗'} | Eq: ${last['equity']:>10.0f}")

    print("\n[5] Summary")
    print("=" * 60)

    for agent in agents:
        logs = agent.logs
        ret_pct = (logs[-1]["equity"] / logs[0]["equity"] - 1) * 100
        trades = sum(1 for x in logs if x["executed"])
        print(f"\n{agent.name}")
        print(f"  Final Equity : ${logs[-1]['equity']:,.0f}")
        print(f"  Return       : {ret_pct:.2f}%")
        print(f"  Trades       : {trades}")

    print("\n[6] Generating comparison plot…")
    plot_agent_comparison(agents, price_df)

    print("\n✅ FINISHED")


if __name__ == "__main__":
    run()
