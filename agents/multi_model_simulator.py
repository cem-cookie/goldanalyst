import sys
sys.path.insert(0, '/Users/sunsichu/Desktop/VirtualTradingOfficer')

print("[DEBUG] Imports starting...")

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

print("[DEBUG] Basic imports OK")

from agents.models.gpt4o_agent import GPT4oAgent
print("[DEBUG] GPT4oAgent imported")

from agents.models.claude_haiku_agent import ClaudeHaikuAgent
print("[DEBUG] ClaudeHaikuAgent imported")

from agents.models.deepseek_agent import DeepSeekAgent
print("[DEBUG] DeepSeekAgent imported")

print("[DEBUG] All imports OK!")
DATA_ROOT = "data/news"
PRICE_CSV = "data/gold_history.csv"
RESULT_DIR = "results"
PLOT_DIR = os.path.join(RESULT_DIR, "plots")

import numpy as np


def plot_comparison(agents, price_df):
    """Compare three agent equity curves + gold price."""

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Gold price on right axis
    ax2 = ax1.twinx()
    ax2.plot(price_df['date'].sort_values(), price_df.sort_values('date')['gold'],
             linewidth=3, label='Gold Price', color='#FFD700', marker='o', markersize=6)
    ax2.set_ylabel('Gold Price (USD/oz)', fontsize=12, fontweight='bold', color='#FFD700')
    ax2.tick_params(axis='y', labelcolor='#FFD700')

    # Three agent equities on left axis
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    for agent, color in zip(agents, colors):
        log_df = pd.DataFrame(agent.logs)
        log_df['date'] = pd.to_datetime(log_df['date'])
        log_df = log_df.sort_values('date')
        ax1.plot(log_df['date'], log_df['equity'], linewidth=2.5, label=f'{agent.name}',
                 color=color, marker='s', markersize=5, alpha=0.8)

    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Portfolio Equity (USD)', fontsize=12, fontweight='bold')
    ax1.set_title('Multi-Agent Trading Performance Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    # Merge legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11, framealpha=0.95)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, "agent_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison: {out_path}")
    plt.close()


def load_gold_prices_robust(path: str):
    """Load without pandas, return simple dict."""

    prices_data = {
        '2025-10-08': 4043.300048828125,
        '2025-10-09': 3946.300048828125,
        '2025-10-10': 3975.89990234375,

        '2025-10-13': 4108.60009765625,
        '2025-10-14': 4138.7001953125,
        '2025-10-15': 4176.89990234375,
        '2025-10-16': 4280.2001953125,
        '2025-10-17': 4189.89990234375,
        '2025-10-20': 4336.39990234375,
        '2025-10-21': 4087.699951171875,
        '2025-10-22': 4044.39990234375,
        '2025-10-23': 4125.5,
        '2025-10-24': 4118.39990234375,
        '2025-10-27': 4001.89990234375,
        '2025-10-28': 3966.199951171875,
        '2025-10-29': 3983.699951171875,
        '2025-10-30': 4001.300048828125,
        '2025-10-31': 3982.199951171875,
        '2025-11-03': 4000.300048828125,
        '2025-11-04': 3947.699951171875,
        '2025-11-05': 3992.10009765625,
    }

    print(f"Loaded {len(prices_data)} records")
    return prices_data


def ensure_dirs():
    """Create required directories."""
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)


def plot_agent_trades(agent, price_df):
    """Generate price and trade chart for single agent."""
    log_df = pd.DataFrame(agent.logs)
    if log_df.empty:
        print(f"[PLOT] {agent.name}: no logs to plot.")
        return

    if "date" in log_df.columns:
        log_df["date"] = pd.to_datetime(log_df["date"])
    else:
        log_df["date"] = pd.to_datetime(price_df["date"].values[: len(log_df)])

    price_df2 = price_df.copy()
    price_df2 = price_df2[["date", "gold"]].rename(columns={"gold": "price_close"})
    merged = pd.merge(price_df2, log_df[["date", "action", "amount_oz", "executed", "price", "equity", "position_oz"]],
                      on="date", how="left")

    buys = merged[(merged["action"] == "BUY") & (merged["executed"] == True)]
    sells = merged[(merged["action"] == "SELL") & (merged["executed"] == True)]

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Top axis: price
    ax1.plot(merged["date"], merged["price_close"], linewidth=2.5, label="Gold Close", color="#2c3e50")
    ax1.fill_between(merged["date"], merged["price_close"].min(), merged["price_close"], alpha=0.1, color="#2c3e50")

    if not buys.empty:
        ax1.scatter(buys["date"], buys["price_close"], marker="^", s=200, label="BUY",
                    color="#2ecc71", zorder=5, edgecolors="darkgreen", linewidth=1.5)
        # Label buy quantity
        for _, row in buys.iterrows():
            ax1.annotate(f"{row['amount_oz']:.1f}oz",
                         xy=(row['date'], row['price_close']),
                         xytext=(0, 10), textcoords='offset points',
                         ha='center', fontsize=8, color='darkgreen', fontweight='bold')

    if not sells.empty:
        ax1.scatter(sells["date"], sells["price_close"], marker="v", s=200, label="SELL",
                    color="#e74c3c", zorder=5, edgecolors="darkred", linewidth=1.5)
        # Label sell quantity
        for _, row in sells.iterrows():
            ax1.annotate(f"{row['amount_oz']:.1f}oz",
                         xy=(row['date'], row['price_close']),
                         xytext=(0, -20), textcoords='offset points',
                         ha='center', fontsize=8, color='darkred', fontweight='bold')

    ax1.set_title(f"{agent.name} - Gold Price & Trading Signals", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Date", fontsize=11)
    ax1.set_ylabel("Gold Price (USD/oz)", fontsize=11)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    # Right axis: equity
    ax2 = ax1.twinx()
    ax2.plot(merged["date"], merged["equity"], linestyle="--", linewidth=2, label="Equity", color="#27ae60")
    ax2.set_ylabel("Portfolio Equity (USD)", fontsize=11, color="#27ae60")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", ncol=2)

    out_path = os.path.join(PLOT_DIR, f"{agent.name}_trades.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def run():
    ensure_dirs()

    print("\n" + "=" * 80)
    print(" GOLD TRADING BACKTESTER - Multi-Model Comparison")
    print("=" * 80)

    # Load gold price
    print("\n[STEP 1] Loading gold price history...")
    all_prices_dict = load_gold_prices_robust(PRICE_CSV)

    # Convert to DataFrame (separate this time)
    print("[DEBUG] Converting to DataFrame...")
    rows = []
    for date_str, price in all_prices_dict.items():
        rows.append({
            'date_str': date_str,
            'gold': price,
            'date': pd.Timestamp(date_str)
        })

    price_df = pd.DataFrame(rows)
    print(f"DataFrame created: {len(price_df)} records")

    # Get dates from news directory
    print("\n[STEP 2] Finding analysis dates in data/news/...")
    have_days = sorted(
        os.path.basename(p.rstrip("/"))
        for p in glob.glob(os.path.join(DATA_ROOT, "*"))
        if os.path.isdir(p)
    )

    if not have_days:
        print("No analysis directories found in data/news/")
        return

    print(f"Found {len(have_days)} analysis dates")

    # Only keep prices for matching dates
    print("\n[STEP 3] Matching prices with analysis dates...")
    price_df = price_df[price_df["date_str"].isin(have_days)].copy().sort_values("date")

    if price_df.empty:
        print("No matching dates")
        return

    print(f"Matched {len(price_df)} days")

    # Initialize agents
    print("\n[STEP 4] Initializing agents...")
    agents = [
        GPT4oAgent("GPT-4o", api_key=os.getenv("OPENAI_API_KEY"), initial_cash=100_000, max_alloc=0.5, fee_bps=10),
        ClaudeHaikuAgent("Claude-Haiku", api_key=os.getenv("ANTHROPIC_API_KEY"), initial_cash=100_000, max_alloc=0.5,
                         fee_bps=10),
        DeepSeekAgent("DeepSeek", api_key=os.getenv("DEEPSEEK_API_KEY"), initial_cash=100_000, max_alloc=0.5,
                       fee_bps=10),
    ]
    print(f"Initialized {len(agents)} agents")

    # Backtest
    print("\n[STEP 5] Running backtest...")
    print("=" * 80)

    for idx, (_, row) in enumerate(price_df.iterrows(), 1):
        d = row["date_str"]
        price = float(row["gold"])

        analysis_path = os.path.join(DATA_ROOT, d, "market_analysis.txt")
        if os.path.exists(analysis_path):
            with open(analysis_path, "r", encoding="utf-8") as f:
                market_summary = f.read()
        else:
            market_summary = "No analysis available."

        print(f"\n[Day {idx:2d}] {d} | Price: ${price:7.2f}")

        for agent in agents:
            decision = agent.decide(market_summary, price)
            agent.execute(decision, price, d)

            # Get latest record from agent.logs
            last_log = agent.logs[-1]

            action = last_log["action"]
            amount = last_log["amount_oz"]
            executed = last_log["executed"]
            eq = last_log["equity"]
            status = "Y" if executed else "N"

            print(f"  {agent.name:<15} {action:5s} {amount:6.2f}oz {status} | Eq: ${eq:>10,.0f}")

    print("=" * 80)

    # Summary
    print("\n[STEP 6] Performance Summary")
    print("=" * 80)

    for agent in agents:
        final_eq = agent.logs[-1]["equity"]
        initial_eq = agent.logs[0]["equity"]
        dd = agent.logs[-1]["max_drawdown"]
        trades = sum(1 for x in agent.logs if x.get("executed"))
        ret_pct = ((final_eq / initial_eq) - 1) * 100

        print(f"\n{agent.name}")
        print(f"  Return: {ret_pct:7.2f}% | Max DD: {dd * 100:7.2f}% | Trades: {trades:2d} | Final: ${final_eq:,.0f}")

    # Plotting
    print("\n[STEP 7] Generating plots...")
    for agent in agents:
        plot_agent_trades(agent, price_df)

    # Plot agent comparison
    print("\n[STEP 8] Plotting agent comparison...")
    plot_comparison(agents, price_df)

    print("\nComplete!")


if __name__ == "__main__":
    run()