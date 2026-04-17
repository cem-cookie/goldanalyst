# scripts/make_mock_trades.py
# -*- coding: utf-8 -*-
import os, json, math, datetime as dt
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
TRADES_CSV = DATA_DIR / "trades_history.csv"
LEDGER_JSON = DATA_DIR / "ledger.json"
PRICE_CSV = DATA_DIR / "gold_history.csv"  # 若存在就用真实价格

RNG = np.random.default_rng(20251106)

def load_price_series(days=120):
    if PRICE_CSV.exists():
        df = pd.read_csv(PRICE_CSV)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values("date").tail(days)
        return df[["date", "gold"]].rename(columns={"gold": "price"}).reset_index(drop=True)
    # fallback: 生成一段金价
    end = dt.date.today()
    start = end - dt.timedelta(days=days + 10)
    dates = pd.bdate_range(start, end)
    steps = RNG.normal(0.00025, 0.0075, len(dates))
    price = 1900 * np.exp(np.cumsum(steps))
    return pd.DataFrame({"date": dates, "price": price})

def simulate_trades(price_df, init_cash=100_000.0, fee_per_trade=2.0):
    """
    简单规则：
      - 随机决定是否交易（~30% 的天有交易）
      - 买入/卖出数量：0.5 ~ 3.0 盎司
      - 不允许做空（持仓>=0），卖出最多卖到 0
    字段：
      date, side, qty, price, notional, fee, realized_pnl, position_qty, avg_cost,
      cash, equity, unrealized_pnl, total_pnl
    """
    rows = []
    cash = init_cash
    pos = 0.0
    avg_cost = 0.0
    realized_pnl = 0.0

    for _, r in price_df.iterrows():
        d = pd.to_datetime(r["date"]).date()
        p = float(r["price"])
        action_flag = RNG.random() < 0.30  # 今天是否交易
        side = "HOLD"
        qty = 0.0
        fee = 0.0
        trade_realized = 0.0

        if action_flag:
            # 50% 买 / 50% 卖（若无仓位则强制买）
            if pos <= 0.0:
                side = "BUY"
            else:
                side = "BUY" if RNG.random() < 0.5 else "SELL"

            qty = float(np.round(RNG.uniform(0.5, 3.0), 2))  # 盎司
            if side == "SELL":
                qty = min(qty, pos)  # 不做空
                if qty < 1e-6:
                    side = "HOLD"

        # 执行
        if side == "BUY":
            notional = qty * p
            fee = fee_per_trade
            if cash >= notional + fee:
                # 加权平均成本
                new_pos = pos + qty
                if new_pos > 0:
                    avg_cost = (pos * avg_cost + qty * p) / new_pos
                pos = new_pos
                cash -= notional + fee
            else:
                side = "HOLD"  # 现金不够就不买
                qty = 0.0
                fee = 0.0
                notional = 0.0

        elif side == "SELL":
            notional = qty * p
            fee = fee_per_trade if qty > 0 else 0.0
            if qty > 0:
                # 实现盈亏 = (卖价-均价)*数量 - 费用(费用计入现金流，这里单独列 fee)
                trade_realized = (p - avg_cost) * qty
                pos -= qty
                # 若清空仓位，均价重置
                if pos <= 1e-9:
                    avg_cost = 0.0
                cash += notional - fee
                realized_pnl += trade_realized
            else:
                notional = 0.0
                fee = 0.0
        else:
            notional = 0.0

        # 估值
        unrealized = (p - avg_cost) * pos if pos > 0 else 0.0
        equity = cash + pos * p
        total_pnl = realized_pnl + unrealized

        rows.append({
            "date": d.isoformat(),
            "side": side,
            "qty": round(qty, 2),
            "price": round(p, 2),
            "notional": round(notional, 2),
            "fee": round(fee, 2),
            "realized_pnl": round(trade_realized, 2),
            "position_qty": round(pos, 4),
            "avg_cost": round(avg_cost, 2),
            "cash": round(cash, 2),
            "equity": round(equity, 2),
            "unrealized_pnl": round(unrealized, 2),
            "total_pnl": round(total_pnl, 2),
        })

    df = pd.DataFrame(rows)
    return df

def summarize(df):
    # 仅统计有实际买/卖的行
    trades = df[df["side"].isin(["BUY", "SELL"])].copy()
    n_trades = len(trades)
    sells = trades[trades["side"] == "SELL"].copy()
    gross_profit = float(np.clip(sells["realized_pnl"], 0, None).sum())
    gross_loss = float(np.clip(sells["realized_pnl"], None, 0).sum())
    net_realized = float(sells["realized_pnl"].sum())
    win_rate = float((sells["realized_pnl"] > 0).mean()) if len(sells) else 0.0

    # 资金曲线
    eq = df["equity"].astype(float).values
    peak = -np.inf
    drawdowns = []
    for v in eq:
        peak = max(peak, v)
        drawdowns.append(0 if peak == 0 else (v - peak) / peak)
    max_dd = float(np.min(drawdowns)) if drawdowns else 0.0

    # 粗略日收益用于夏普
    ret = pd.Series(eq).pct_change().dropna()
    sharpe = float(np.sqrt(252) * (ret.mean() / (ret.std() + 1e-9))) if len(ret) > 3 else 0.0

    summary = {
        "trading_days": int(df["date"].nunique()),
        "num_trades": int(n_trades),
        "num_sells_closed": int(len(sells)),
        "win_rate": round(win_rate * 100, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "net_realized": round(net_realized, 2),
        "final_equity": float(df.iloc[-1]["equity"]),
        "final_cash": float(df.iloc[-1]["cash"]),
        "final_position": float(df.iloc[-1]["position_qty"]),
        "unrealized_pnl": float(df.iloc[-1]["unrealized_pnl"]),
        "total_pnl": float(df.iloc[-1]["total_pnl"]),
        "max_drawdown": round(max_dd * 100, 2),
        "sharpe_like": round(sharpe, 3),
    }
    return summary

def main():
    price_df = load_price_series(days=120)
    trades_df = simulate_trades(price_df)
    trades_df.to_csv(TRADES_CSV, index=False, encoding="utf-8")

    summary = summarize(trades_df)
    LEDGER_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✓ Mock trades written to {TRADES_CSV}")
    print(f"✓ Ledger summary written to {LEDGER_JSON}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
