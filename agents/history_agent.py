# agents/history_agent.py
# -*- coding: utf-8 -*-
import json
import pandas as pd
from pathlib import Path

class HistoryAgent:
    def __init__(self, trades_csv="data/trades_history.csv", ledger_json="data/ledger.json"):
        self.trades_csv = Path(trades_csv)
        self.ledger_json = Path(ledger_json)

    def load(self):
        if not self.trades_csv.exists():
            raise FileNotFoundError(f"Trades file not found: {self.trades_csv}")
        df = pd.read_csv(self.trades_csv)
        # type compatibility
        for col in [
            "qty","price","notional","fee","realized_pnl","position_qty",
            "avg_cost","cash","equity","unrealized_pnl","total_pnl"
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def summary(self):
        if self.ledger_json.exists():
            return json.loads(self.ledger_json.read_text(encoding="utf-8"))
        # fallback lightweight summary if ledger.json is missing
        df = self.load()
        sells = df[df["side"] == "SELL"]
        net_realized = float(sells["realized_pnl"].sum())
        win_rate = float((sells["realized_pnl"] > 0).mean()) * 100 if len(sells) > 0 else 0.0
        return {
            "num_trades": int(len(df[df["side"].isin(["BUY","SELL"])])),
            "win_rate": round(win_rate, 2),
            "net_realized": round(net_realized, 2),
            "final_equity": float(df.iloc[-1]["equity"]),
            "final_position": float(df.iloc[-1]["position_qty"]),
            "unrealized_pnl": float(df.iloc[-1]["unrealized_pnl"]),
            "total_pnl": float(df.iloc[-1]["total_pnl"]),
        }

    def report_text(self, as_html=False):
        s = self.summary()
        parts = []
        parts.append(f"• Total trades: {s.get('num_trades', '—')}, Win rate: {s.get('win_rate', '—')}%.")
        if "net_realized" in s:
            parts.append(f"• Realized P&L: ${s['net_realized']:.2f}.")
        parts.append(f"• Current equity: ${s.get('final_equity', '—'):.2f}.")
        parts.append(
            f"• Current position: {s.get('final_position', '—')} oz, Unrealized P&L: ${s.get('unrealized_pnl', '—'):.2f}.")
        parts.append(f"• Total P&L (incl. unrealized): ${s.get('total_pnl', '—'):.2f}.")
        conclusion = "Overall performance is stable" if s.get("total_pnl",
                                                              0) >= 0 else "Consider managing drawdowns and trade frequency"
        parts.append(f"Conclusion: {conclusion}.")
        if as_html:
            return "<br>".join(parts)
        return "\n".join(parts)

    # 在类里新增这个方法（或把原来的 report_text 改成这样）
    def report_markdown(self):
        s = self.summary()
        lines = [
            f"- **Total trades:** {s.get('num_trades', '—')}, **Win rate:** {s.get('win_rate', '—')}%.",
            f"- **Realized P&L:** ${s.get('net_realized', 0):,.2f}.",
            f"- **Current equity:** ${s.get('final_equity', '—'):,.2f}.",
            f"- **Current position:** {s.get('final_position', '—')} oz, **Unrealized P&L:** ${s.get('unrealized_pnl', '—'):,.2f}.",
            f"- **Total P&L (incl. unrealized):** ${s.get('total_pnl', '—'):,.2f}.",
        ]
        conclusion = ("Overall performance is stable"
                      if s.get("total_pnl", 0) >= 0
                      else "Consider managing drawdowns and trade frequency")
        lines.append(f"**Conclusion:** {conclusion}.")
        return "\n".join(lines)
