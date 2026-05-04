"""
agents/simulate_ds_comparison.py

Comparison:
1) DeepSeek (base)
2) DeepSeek-LoRA (fine-tuned, EAS)
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from agents.models.deepseek_agent import DeepSeekAgent
from agents.models.deepseek_LoRA_agent import DeepSeekLoRAAgent

DATA_ROOT = "data/news"
PRICE_CSV = "data/gold_history.csv"
RESULT_DIR = "results"
PLOT_DIR = os.path.join(RESULT_DIR, "plots")


# Your fine-tuned model's EAS endpoint
EAS_URL = "http://1533366639129314.cn-beijing.pai-eas.aliyuncs.com/api/predict/quickstart_deploy_20251225_1vxd"
EAS_TOKEN = "MjY0ZjEwNDFiODk4ZGFiOTEzYjMwZWNlNTRmNzRlY2QxZDMyMzRmMQ=="  # Change this


def ensure_dirs():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)


def load_prices(csv_path):
    """
    Robust CSV loader:
    - Supports with/without header
    - Auto-detect date and price columns
    - Normalize to column names: date, price
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]

    first_line = lines[0]
    first_col = first_line.split(",")[0].lower()
    has_header = first_col in ["date", "day", "time"]

    if has_header:
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(csv_path, header=None, names=["date", "price"])

    # Normalize date column
    if "date" not in df.columns:
        date_col = None
        for c in df.columns:
            if "date" in c.lower():
                date_col = c
                break
        if date_col is None:
            date_col = df.columns[0]
        df = df.rename(columns={date_col: "date"})

    # Normalize price column
    if "price" not in df.columns:
        price_col = None
        for c in df.columns:
            cl = c.lower()
            if cl in ["price", "close", "gold", "xauusd", "xau", "value"]:
                price_col = c
                break
        if price_col is None:
            non_date_cols = [c for c in df.columns if c != "date"]
            if len(non_date_cols) == 0:
                raise ValueError(f"[load_prices] Cannot find price column. Columns={df.columns.tolist()}")
            price_col = non_date_cols[0]
        df = df.rename(columns={price_col: "price"})

    df = df[["date", "price"]].dropna()
    df = df[df["date"].astype(str).str.match(r"\d{4}-\d{2}-\d{2}")]
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df = df.dropna(subset=["date"])
    df["price"] = df["price"].astype(float)

    return df.sort_values("date")


def plot_agents(agents, price_df):
    fig, ax1 = plt.subplots(figsize=(13, 7))
    ax2 = ax1.twinx()

    ax2.plot(price_df["date"], price_df["price"], linewidth=2.0, label="Gold Price")

    for agent in agents:
        log = pd.DataFrame(agent.logs)
        log["date"] = pd.to_datetime(log["date"])
        ax1.plot(log["date"], log["equity"], linewidth=2.0, label=agent.name)

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Portfolio Equity")
    ax2.set_ylabel("Gold Price")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.grid(alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "deepseek_comparison.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved plot -> {out}")


def run():
    ensure_dirs()
    prices = load_prices(PRICE_CSV)

    news_days = sorted(
        os.path.basename(p)
        for p in glob.glob(f"{DATA_ROOT}/*")
        if os.path.isdir(p)
    )

    prices["date_str"] = prices["date"].dt.strftime("%Y-%m-%d")
    prices = prices[prices["date_str"].isin(news_days)].sort_values("date")

    if prices.empty:
        print("No matching days between price and news folders!")
        return

    base_agent = DeepSeekAgent(
        name="DeepSeek-Base",
        initial_cash=100_000,
        max_alloc=0.5,
        fee_bps=10,
    )

    lora_agent = DeepSeekLoRAAgent(
        name="DeepSeek-LoRA",
        eas_url=EAS_URL,
        eas_token=EAS_TOKEN,
        initial_cash=100_000,
        max_alloc=0.5,
        fee_bps=10,
    )

    agents = [base_agent, lora_agent]

    for _, row in prices.iterrows():
        date = row["date_str"]
        price = row["price"]

        path = f"{DATA_ROOT}/{date}/market_analysis.txt"
        summary = open(path, "r", encoding="utf-8").read() if os.path.exists(path) else "No analysis."

        print(f"\n[{date}] price={price:.2f}")

        for a in agents:
            d = a.decide(summary, price)
            a.execute(d, price, date)
            last = a.logs[-1]
            print(f"  {a.name:<14} {last['action']:>5} {last['amount_oz']:>6.2f} oz "
                  f"{'Y' if last.get('executed') else 'N'} | Eq: {last['equity']:.0f}")

    for a in agents:
        ret = (a.logs[-1]["equity"] / a.logs[0]["equity"] - 1) * 100
        trades = sum(1 for x in a.logs if x.get("executed"))
        print(f"\n{a.name}")
        print(f"  Return: {ret:.2f}% | Trades: {trades} | Final Eq: {a.logs[-1]['equity']:.0f}")

    plot_agents(agents, prices)


if __name__ == "__main__":
    run()