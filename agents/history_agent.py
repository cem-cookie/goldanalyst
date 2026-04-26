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
        df = pd.read_csv(self.trades_csv, encoding="utf-8")
        for col in [
            "qty","price","notional","fee","realized_pnl","position_qty",
            "avg_cost","cash","equity","unrealized_pnl","total_pnl"
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _safe_float(self, val):
        """Convert value to float safely, returning None if not numeric."""
        if val is None:
            return None
        try:
            result = float(val)
            return result if pd.notna(result) else None
        except (TypeError, ValueError):
            return None

    def _fmt_currency(self, val):
        """Format numeric value as currency, returning '—' for None."""
        if val is None:
            return "—"
        try:
            return f"${val:.2f}"
        except (TypeError, ValueError):
            return "—"

    def _fmt_number(self, val):
        """Format numeric value, returning '—' for None."""
        if val is None:
            return "—"
        try:
            return f"{val:.2f}"
        except (TypeError, ValueError):
            return "—"

    def summary(self):
        if self.ledger_json.exists():
            return json.loads(self.ledger_json.read_text(encoding="utf-8"))
        df = self.load()

        if len(df) == 0:
            raise ValueError("No trades to analyze")

        sells = df[df["side"] == "SELL"]
        net_realized = float(sells["realized_pnl"].sum()) if len(sells) > 0 else 0.0
        win_rate = float((sells["realized_pnl"] > 0).mean()) * 100 if len(sells) > 0 else 0.0

        last_row = df.iloc[-1]
        return {
            "num_trades": int(len(df[df["side"].isin(["BUY","SELL"])])),
            "win_rate": round(win_rate, 2),
            "net_realized": round(net_realized, 2),
            "final_equity": self._safe_float(last_row.get("equity")),
            "final_position": self._safe_float(last_row.get("position_qty")),
            "unrealized_pnl": self._safe_float(last_row.get("unrealized_pnl")),
            "total_pnl": self._safe_float(last_row.get("total_pnl")),
        }

    def report_text(self, as_html=False):
        s = self.summary()
        parts = []
        parts.append(f"* Total trades: {s.get('num_trades', 0)}, Win rate: {s.get('win_rate', 0)}%.")
        if "net_realized" in s:
            parts.append(f"* Realized P&L: {self._fmt_currency(s.get('net_realized'))}.")
        parts.append(f"* Current equity: {self._fmt_currency(s.get('final_equity'))}.")
        parts.append(
            f"* Current position: {self._fmt_number(s.get('final_position'))} oz, Unrealized P&L: {self._fmt_currency(s.get('unrealized_pnl'))}.")
        parts.append(f"* Total P&L (incl. unrealized): {self._fmt_currency(s.get('total_pnl'))}.")
        conclusion = "Overall performance is stable" if s.get("total_pnl", 0) >= 0 else "Consider managing drawdowns and trade frequency"
        parts.append(f"Conclusion: {conclusion}.")
        if as_html:
            return "<br>".join(parts)
        return "\n".join(parts)

    def report_markdown(self):
        s = self.summary()
        lines = [
            f"- **Total trades:** {s.get('num_trades', 0)}, **Win rate:** {s.get('win_rate', 0)}%.",
            f"- **Realized P&L:** {self._fmt_currency(s.get('net_realized'))}.",
            f"- **Current equity:** {self._fmt_currency(s.get('final_equity'))}.",
            f"- **Current position:** {self._fmt_number(s.get('final_position'))} oz, **Unrealized P&L:** {self._fmt_currency(s.get('unrealized_pnl'))}.",
            f"- **Total P&L (incl. unrealized):** {self._fmt_currency(s.get('total_pnl'))}.",
        ]
        conclusion = ("Overall performance is stable"
                      if s.get("total_pnl", 0) >= 0
                      else "Consider managing drawdowns and trade frequency")
        lines.append(f"**Conclusion:** {conclusion}.")
        return "\n".join(lines)
