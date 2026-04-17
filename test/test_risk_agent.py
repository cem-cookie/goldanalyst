import os
import sys
from agents.risk_agent import RiskAgent


def pick_decision_path():
    # Common paths in priority order
    candidates = ["data/trading_decision.json", "trading_decision.json"]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def main():
    decision_path = pick_decision_path()
    if not decision_path:
        print("[ERROR] trading_decision.json not found. Please run Actions to generate decision first.")
        sys.exit(1)

    out_path = "data/risk_report.json"
    print(f"[INFO] Using decision file → {decision_path}")

    agent = RiskAgent(
        decision_path=decision_path,
        out_path=out_path,
        openai_api_key=os.getenv("OPENAI_API_KEY")  # If not set, will automatically use heuristic fallback
    )

    report = agent.run()
    if not report:
        print("[ERROR] RiskAgent did not return a report.")
        sys.exit(2)

    # —— Console summary ——
    summary = report.get("summary", {})
    print("\n=== RISK OVERVIEW ===")
    print(f"Asset : {report.get('asset','—')}")
    print(f"Evaluated at : {report.get('evaluated_at','—')}")
    print(f"Portfolio risk: {summary.get('portfolio_risk','—')}")
    if summary.get("comment"):
        print(f"Comment : {summary['comment']}")

    print("\n--- PER-STRATEGY ---")
    for it in report.get("items", []):
        print(f"\n[Strategy #{it.get('id','?')}]")
        print(f" Approval : {it.get('approval','—')} ({int(it.get('approval_score',0))}/100)")
        print(f" Risk level : {it.get('risk_level','—')}")
        if it.get("key_risks"):
            print(" Key risks :")
            for r in it["key_risks"]:
                print(f" - {r}")
        if it.get("mitigations"):
            print(" Mitigations:")
            for m in it["mitigations"]:
                print(f" - {m}")
        if it.get("notes"):
            print(f" Notes : {it['notes']}")

    print(f"\n[INFO] Risk report saved → {out_path}\n")


if __name__ == "__main__":
    main()