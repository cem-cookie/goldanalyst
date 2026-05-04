# -*- coding: utf-8 -*-
import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import json
import datetime as dt
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import yfinance
from cryptography.fernet import Fernet
from pathlib import Path
from datetime import datetime as dt_datetime

# === Local imports ===
from agents.data_agent import DataAgent
from agents.trading_agent import TradingAgent
from agents.risk_agent import RiskAgent
from mock_trades import main as make_mock
from dashboard.auto_scheduler import get_scheduler
from dashboard.auto_archive import get_archiver
from dashboard.pipeline_runner import get_pipeline

# ---------- Page ----------
st.set_page_config(page_title="Gold AI Arena Dashboard", layout="wide")

# ---------- CSS ----------
css_file = Path(__file__).with_name("style.css")
if css_file.exists():
    st.markdown(f"<style>{css_file.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ---------- State ----------
st.session_state.setdefault("timeframe", "3 years")
st.session_state.setdefault("strategy", "Swing")
st.session_state.setdefault("sources", ["investing.com", "yahoo"])
st.session_state.setdefault("active_tool", None)
st.session_state.setdefault("investment", "Active")
st.session_state.setdefault("target_profit", 0.1)
st.session_state.setdefault("show_logs", True)
# Automation state
st.session_state.setdefault("auto_mode", True)
st.session_state.setdefault("refresh_interval", 60)
st.session_state.setdefault("is_paused", False)
st.session_state.setdefault("last_update_time", None)
st.session_state.setdefault("pipeline_running", False)
# Position sizing defaults
st.session_state.setdefault("risk_percent", 2.0)
st.session_state.setdefault("position_size_oz", None)  # None = auto-calculate
# Trade mode defaults
st.session_state.setdefault("trade_mode", "Buy")  # "Buy" or "Sell"
st.session_state.setdefault("latest_price", None)


# ---------- Helper Functions ----------
def _resolve_api_key(user_provided_key: str = None) -> str | None:
    """
    Resolve API key with fallback chain: user input -> env var -> st.secrets.
    Priority: user_provided_key > os.getenv > st.secrets > None
    """
    if user_provided_key:
        return user_provided_key
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key
    try:
        if "openai_api_key" in st.secrets:
            return st.secrets["openai_api_key"]
    except Exception:
        pass
    return None


def _get_user_api_key() -> str | None:
    """
    Retrieve decrypted user-provided API key from session state.
    Returns None if no valid key can be decrypted.
    """
    if st.session_state.get('api_key_enc') and st.session_state.get('fernet_key'):
        f = Fernet(st.session_state['fernet_key'].encode())
        return f.decrypt(st.session_state['api_key_enc'].encode()).decode()
    return None


def fetch_latest_gold_price() -> float | None:
    """
    Fetch the latest gold price using yfinance.
    Tries multiple ticker symbols for gold.
    Returns None if the price cannot be retrieved.
    """
    gold_tickers = ["GC=F", "XAUUSD=X", "GLD"]
    for ticker_symbol in gold_tickers:
        try:
            ticker = yfinance.Ticker(ticker_symbol)
            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                latest = hist["Close"].iloc[-1]
                return round(float(latest), 2)
        except Exception:
            continue
    print(f"[WARN] Failed to fetch latest gold price from any ticker")
    return None


# ---------- Data ----------
@st.cache_data(ttl=300)  # 5 minute cache for auto-refresh
def make_gold_series():
    """Load real gold price data from CSV file."""
    csv_path = "data/gold_history.csv"

    # If file does not exist
    if not os.path.exists(csv_path):
        st.warning("Gold price data not found. Using dummy data. Run data update first.")
        end = dt.date.today()
        start = end - dt.timedelta(days=365 * 3 + 10)
        dates = pd.date_range(start, end, freq="D")
        rng = np.random.default_rng(2025)
        steps = rng.normal(0.0003, 0.008, len(dates))
        price = 1800 * np.exp(np.cumsum(steps))
        return pd.DataFrame({"date": dates, "gold": price})

    # Read real data
    df = pd.read_csv(csv_path, header=0)  # header
    df["date"] = pd.to_datetime(df["date"], errors='coerce')  # errors='coerce' will convert unparseable dates to NaT
    df = df.dropna(subset=["date"])  # Remove rows with invalid dates

    return df

def filter_by_range(df: pd.DataFrame, key: str):
    """Slice time range by label."""
    end = df["date"].max().date()
    if key == "3 years":
        start = end - dt.timedelta(days=365 * 3)
    elif key == "1 year":
        start = end - dt.timedelta(days=365)
    elif key == "6 months":
        start = end - dt.timedelta(days=30 * 6)
    else:
        start = end - dt.timedelta(days=30 * 3)
    return df[df["date"].dt.date.between(start, end)]


def price_chart(df: pd.DataFrame):
    """Altair line chart for gold with MA and Bollinger Bands."""
    df = df.copy()

    # Calculate indicators
    df['MA20'] = df['gold'].rolling(window=20).mean()
    df['MA50'] = df['gold'].rolling(window=50).mean()

    # Bollinger Bands (20-day, 2 std)
    df['BB_std'] = df['gold'].rolling(window=20).std()
    df['BB_upper'] = df['MA20'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['MA20'] - (df['BB_std'] * 2)

    # Base chart
    base = alt.Chart(df).encode(
        x=alt.X("date:T", title=None)
    )

    # Bollinger Bands (shaded area)
    bb_area = base.mark_area(opacity=0.15, color='lightblue').encode(
        y=alt.Y("BB_upper:Q", scale=alt.Scale(zero=False)),
        y2=alt.Y2("BB_lower:Q")
    )

    # Price line
    price_line = base.mark_line(color='#1f77b4', strokeWidth=2).encode(
        y=alt.Y("gold:Q", title=None, scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip("gold:Q", title="Price", format=".2f"),
            alt.Tooltip("MA20:Q", title="MA20", format=".2f"),
            alt.Tooltip("MA50:Q", title="MA50", format=".2f"),
            alt.Tooltip("BB_upper:Q", title="BB Upper", format=".2f"),
            alt.Tooltip("BB_lower:Q", title="BB Lower", format=".2f")
        ]
    )

    # Moving average lines
    ma20_line = base.mark_line(color='orange', strokeWidth=1.5, strokeDash=[5, 5]).encode(
        y=alt.Y("MA20:Q")
    )

    ma50_line = base.mark_line(color='red', strokeWidth=1.5, strokeDash=[3, 3]).encode(
        y=alt.Y("MA50:Q")
    )

    # Combine (BB area first, then lines on top)
    return (bb_area + ma20_line + ma50_line + price_line).properties(
        height=280,
        background="transparent"
    ).configure_view(
        fill="transparent"
    )


def run_news_pipeline(agent: DataAgent, sources: list[str], out, limit: int = 10):
    """Drive the end-to-end news flow with incremental logs; only final summary remains."""

    def log(line: str):
        out.markdown(f"<div class='log-line'>• {line}</div>", unsafe_allow_html=True)

    json_path = "gold_news.json"

    out.markdown("<div class='log-line'>Starting news search...</div>", unsafe_allow_html=True)
    log(f"Sources selected: {', '.join(sources) if sources else 'None'}")

    log("Collecting feeds...")
    all_news = []
    if "yahoo" in sources:
        all_news += agent.get_yahoo_rss_news(limit)
    if "MetalsDaily" in sources:
        all_news += agent.get_metalsdaily_news(limit)
    if "investing.com" in sources:
        all_news += agent.get_investing_news(limit)

    if not all_news:
        out.markdown("### Gold Market Briefing — No Sources Selected\n\nPlease choose at least one news source.",
                     unsafe_allow_html=True)
        return

    # Renumber items
    for i, n in enumerate(all_news, 1):
        n["id"] = i

    # Save original news
    os.makedirs("data", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_news, f, ensure_ascii=False, indent=2)

    log(f"Collected {len(all_news)} items. Filtering relevant ones via LLM...")

    # One-time processing: filter + fetch full text + calculate quality scores
    filtered_news = agent.filter_and_fetch_fulltext(json_path)

    log("Fetching full text for selected articles... ✓")
    log("Calculating quality scores... ✓")
    log("Summarizing with LLM...")

    summary = agent.analyze_market_news(json_path, min_quality_score=30)

    # Clear logs BEFORE displaying final report
    out.empty()

    title = "Gold Market Briefing — " + dt.date.today().isoformat()
    st.markdown(f"## {title}")
    st.markdown(summary)


def run_risk_pipeline(agent: RiskAgent, out):
    """Drive the end-to-end risk assessment flow with incremental logs; only final report remains."""

    def log(line: str):
        out.markdown(f"<div class='log-line'>• {line}</div>", unsafe_allow_html=True)

    out.markdown("<div class='log-line'>Starting risk assessment...</div>", unsafe_allow_html=True)

    log("Loading trading decision...")
    log("Performing risk analysis via LLM...")

    try:
        agent.run()  # Generate data/risk_report.json
    except FileNotFoundError:
        # Clear logs BEFORE displaying error
        out.empty()
        out.markdown("### Risk Assessment\n\nNo trading decision found. Please run **Actions** first.")
        return

    log("Risk report saved successfully.")

    # Clear logs BEFORE displaying final report
    out.empty()


# ---------- Title ----------
#st.markdown("## Virtual Trading Officer for Gold")
# Title with History button
col1, col2 = st.columns([1, 0.15])

with col1:
    st.markdown("## Virtual Trading Officer for Gold")

with col2:
    if st.button("Trading History", width='stretch', key="btn_history"):
        st.session_state.active_tool = "History"



# ---------- Layout ----------
left, right = st.columns([6, 6], gap="small")

# === LEFT: Text output with border and fixed height ===
with left:
    with st.container(border=True, height=670):  # Fixed height, auto-scroll
        st.markdown("#### Output")
        output_area = st.container()  # Use container instead of empty()

        # === History Panel ===
        if st.session_state.active_tool == "History":
            output_area.empty()  # Clear left Output
            with output_area:
                from pathlib import Path
                from agents.history_agent import HistoryAgent

                st.subheader("Trading History")

                trades_csv = Path("data/trades_history.csv")
                ledger_json = Path("data/ledger.json")

                # No data -> provide one-click mock generation
                if not trades_csv.exists():
                    st.warning("No trading history found.")
                    if st.button("Generate mock trading data", key="gen_mock", width='stretch'):
                        try:
                            # Reuse your earlier script

                            make_mock()
                            st.success("Mock data generated. Click 'Trading History' again if needed.")
                        except Exception as e:
                            st.error(f"Failed to generate mock data: {e}")
                else:
                    # Load and display summary
                    agent = HistoryAgent(trades_csv=str(trades_csv), ledger_json=str(ledger_json))
                    st.markdown("#### Summary")
                    st.markdown(agent.report_markdown())

                    # Optional: recent trades table
                    with st.expander("Show recent trades", expanded=False):
                        df_hist = agent.load()
                        st.dataframe(df_hist.tail(50), width='stretch', height=280)

# === RIGHT: Top — Chart ===
with right:
    with st.container(border=True):
        st.markdown("#### Gold (XAU/USD)")
        tf = st.radio(
            "Time range",
            ["3 years", "1 year", "6 months", "3 months"],
            horizontal=True,
            label_visibility="collapsed",
            index=["3 years", "1 year", "6 months", "3 months"].index(st.session_state.timeframe),
        )
        st.session_state.timeframe = tf
        full_df = make_gold_series()
        df = filter_by_range(full_df, tf)
        st.altair_chart(price_chart(df), width='stretch')

    # === RIGHT: Bottom — Control Panel ===
    with st.container(border=True):
        # Top spacer — padding between Control Panel border and AI Settings
        top_spacer = st.container()
        top_spacer.markdown("<div style='padding-top: 18px;'></div>", unsafe_allow_html=True)

        # AI Settings section
        ai_section = st.container()
        with ai_section:
            # --- AI Model & API Key Settings ---
            with st.expander("🤖 AI Model & API Key Settings", expanded=False):
                # Model selector (default ChatGPT)
                model = st.radio(
                    "Select LLM",
                    ["ChatGPT (OpenAI)", "Claude (Anthropic)"],
                    index=0,
                    key="llm_selector"
                )
                st.session_state['selected_model'] = model

                # API key entry – password masked
                api_key_raw = st.text_input(
                    f"{model.split(' ')[0]} API key",
                    type="password",
                    placeholder="Enter your API key …",
                    key="api_key_input"
                )

                if api_key_raw:
                    # Generate per‑session Fernet key if not present
                    if 'fernet_key' not in st.session_state:
                        st.session_state['fernet_key'] = Fernet.generate_key().decode()
                    f = Fernet(st.session_state['fernet_key'].encode())
                    st.session_state['api_key_enc'] = f.encrypt(api_key_raw.encode()).decode()
                    # Persistent security warning for user-provided keys
                    st.warning(
                        "⚠️ Security Notice: For your protection, we recommend using a "
                        "one-time or restricted API key that cannot access sensitive billing "
                        "information. Do not use your primary API key."
                    )

                # Provider link for convenience
                provider_link = {
                    "ChatGPT (OpenAI)": "https://platform.openai.com/account/api-keys",
                    "Claude (Anthropic)": "https://console.anthropic.com/account/keys"
                }
                st.markdown(
                    f"You can get your API key from → [{provider_link[model]}]({provider_link[model]})",
                    unsafe_allow_html=False
                )

        # === Automation Settings ===
        with st.expander("📡 Automation Settings", expanded=True):
            # Auto/Manual toggle
            col_auto1, col_auto2 = st.columns([1, 1])
            with col_auto1:
                new_auto_mode = st.toggle(
                    "Auto-refresh",
                    value=st.session_state.auto_mode,
                    key="auto_mode_toggle"
                )
                if new_auto_mode != st.session_state.auto_mode:
                    st.session_state.auto_mode = new_auto_mode
                    scheduler = get_scheduler()
                    scheduler.toggle_auto_mode()

            with col_auto2:
                new_paused = st.toggle(
                    "Pause",
                    value=st.session_state.is_paused,
                    key="pause_toggle"
                )
                if new_paused != st.session_state.is_paused:
                    st.session_state.is_paused = new_paused
                    scheduler = get_scheduler()
                    scheduler.toggle_pause()

            # Interval selector
            col_int1, col_int2 = st.columns([1, 2])
            with col_int1:
                st.caption("Refresh interval:")
            with col_int2:
                new_interval = st.select_slider(
                    "Interval",
                    options=[15, 30, 60, 120, 240],
                    value=st.session_state.refresh_interval,
                    format_func=lambda x: f"{x} min",
                    key="interval_slider"
                )
                if new_interval != st.session_state.refresh_interval:
                    st.session_state.refresh_interval = new_interval
                    scheduler = get_scheduler()
                    scheduler.set_interval(new_interval)

            # Status display
            scheduler = get_scheduler()
            status = scheduler.get_status()
            api_info = status.get("api_usage", {})

            # API Usage
            col_api1, col_api2 = st.columns([2, 1])
            with col_api1:
                usage_pct = api_info.get("percentage", 0)
                st.progress(min(usage_pct/100, 1.0))
            with col_api2:
                st.caption(f"API: {api_info.get('current', 0)}/{api_info.get('limit', 1000)}")

            # Next update time
            if status.get("next_run"):
                next_dt = dt_datetime.fromisoformat(status["next_run"])
                now = dt_datetime.now()
                if next_dt > now:
                    remaining = (next_dt - now).total_seconds() / 60
                    st.caption(f"⏱️ Next auto-update: {remaining:.0f} min")
                else:
                    st.caption("⏱️ Update available now")

            # Error alert
            if status.get("should_alert"):
                st.error("⚠️ Multiple errors detected. Check logs.")
                recent_errors = status.get("recent_errors", [])
                if recent_errors:
                    with st.expander("Recent errors", expanded=False):
                        for err in recent_errors[-3:]:
                            st.caption(f"{err.get('step', 'unknown')}: {err.get('error', 'error')[:100]}")

            # Refresh Now button
            if st.button("🔄 Refresh Now", key="btn_refresh_now", width='stretch'):
                with st.spinner("Running full pipeline..."):
                    pipeline = get_pipeline()
                    context = {
                        "strategy": st.session_state.strategy,
                        "investment_level": st.session_state.investment,
                        "buy_price_threshold": st.session_state.buy_price_threshold,
                        "sell_price_threshold": st.session_state.sell_price_threshold,
                        "target_profit": st.session_state.target_profit,
                    }
                    result = pipeline.run_full_pipeline(context)
                    st.session_state.last_update_time = dt_datetime.now().isoformat()

                    if result.get("success"):
                        st.success(f"Pipeline completed: {result.get('steps_completed', 0)}/4 steps")
                    else:
                        st.warning(f"Pipeline blocked: {result.get('blocked_reason', 'Unknown')}")

            # Archive info
            archiver = get_archiver()
            total_size = archiver.get_total_size()
            if total_size > 0:
                st.caption(f"📦 Archives: {total_size:.1f} MB in data/archive/")

        
        # Row: Strategy
        r1c1, r1c2, r1c3, r1c4= st.columns([1, 2, 1, 2])
        with r1c1:
            st.markdown('<div class="inline-label">Strategy</div>', unsafe_allow_html=True)
        with r1c2:
            st.session_state.strategy = st.selectbox(
                "Strategy",
                ["Scalping", "Swing", "Seasonal"],
                index=["Scalping", "Swing", "Seasonal"].index(st.session_state.strategy),
                label_visibility="collapsed",
            )
        with r1c3:
            st.markdown('<div class="inline-label">Investment</div>', unsafe_allow_html=True)
        with r1c4:
            st.session_state.investment = st.selectbox(
                "Investment",
                ["Passive", "Active", "Aggressive"],
                index=["Passive", "Active", "Aggressive"].index(st.session_state.investment),
                label_visibility="collapsed",
            )

        # Row: Trade Price & Direction
        col_trade1, col_trade2 = st.columns([1, 2], gap="small")

        with col_trade1:
            st.markdown('<div class="param-label">📊 Trade Mode</div>', unsafe_allow_html=True)
            # Trade mode: Buy or Sell (default Buy)
            trade_mode = st.radio(
                "Direction",
                ["Buy", "Sell"],
                index=0 if st.session_state.trade_mode == "Buy" else 1,
                key="trade_mode_radio",
                horizontal=True,
                label_visibility="collapsed"
            )
            st.session_state.trade_mode = trade_mode

        with col_trade2:
            # Get latest price for display only (AI recommends entry price based on strategies)
            latest_price = st.session_state.latest_price
            if latest_price is None:
                latest_price = fetch_latest_gold_price()
                st.session_state.latest_price = latest_price

            # Display live price as reference only (AI determines entry price)
            if latest_price:
                st.markdown(f'<div class="param-label">📊 Current Market Price</div>', unsafe_allow_html=True)
                st.markdown(f"### ${latest_price:,.2f}")
            else:
                st.markdown('<div class="param-label">📊 Market Price</div>', unsafe_allow_html=True)
                st.markdown("### --")

            # Row: Actions (Refresh Price)
            col_price_btn1, col_price_btn2 = st.columns([1, 1], gap="small")
            with col_price_btn1:
                if st.button("🔄 Refresh Price", key="btn_load_price", use_container_width=True,
                          help="Refresh latest market price"):
                    fetched = fetch_latest_gold_price()
                    if fetched:
                        st.session_state.latest_price = fetched
                    st.rerun()
            with col_price_btn2:
                st.caption("AI will recommend entry price based on strategies")

        # Row: Position & Target Profit
        ps1, ps2, ps3 = st.columns(3, gap="small")

        with ps1:
            st.markdown('<div class="param-label">⚡ Risk %</div>', unsafe_allow_html=True)
            risk_pct = float(st.session_state.risk_percent)
            st.session_state.risk_percent = st.number_input(
                "Risk %",
                min_value=0.1,
                max_value=5.0,
                value=risk_pct,
                step=0.1,
                format="%.1f",
                label_visibility="collapsed",
                key="risk_percent_input",
                help="Percentage of account balance to risk per trade (default: 2%)"
            )
            st.caption("Risk % per trade")

        with ps2:
            st.markdown('<div class="param-label">📏 Position (oz)</div>', unsafe_allow_html=True)
            pos_size = st.session_state.position_size_oz
            st.session_state.position_size_oz = st.number_input(
                "Position oz",
                min_value=0.01,
                max_value=100.0,
                value=pos_size if pos_size else 1.0,
                step=0.1,
                format="%.2f",
                label_visibility="collapsed",
                key="position_size_input",
                help="Position size in ounces. Leave empty for auto-calculate based on risk % and confidence."
            )
            st.caption("Leave empty = auto size")

        with ps3:
            st.markdown('<div class="param-label">💵 Account</div>', unsafe_allow_html=True)
            if "account_balance" not in st.session_state:
                st.session_state.account_balance = 100000.0
            acct_bal = float(st.session_state.account_balance)
            st.session_state.account_balance = st.number_input(
                "Balance",
                min_value=100.0,
                max_value=10000000.0,
                value=acct_bal,
                step=1000.0,
                format="%.0f",
                label_visibility="collapsed",
                key="account_balance_input",
                help="Total account balance in USD for position sizing"
            )
            st.caption("Account balance (USD)")

        # Action buttons
        c1, c2, c3 = st.columns(3, gap="small")
        with c1:
            news_type = "primary" if st.session_state.active_tool == "News" else "secondary"
            if st.button("📰 News", key="btn_news", width='stretch', type=news_type):
                st.session_state.active_tool = "News"
                output_area.empty()  # Clean Container

                with output_area:
                    # Decrypt user‑provided API key for LLM usage
                    model = st.session_state.get('selected_model','ChatGPT (OpenAI)')
                    if model != 'ChatGPT (OpenAI)':
                        st.error('Claude model is not supported for news collection currently.')
                        st.stop()
                    api_key = _resolve_api_key(_get_user_api_key())
                    if not api_key:
                        st.error("No API key available. Please enter your OpenAI API key above, or set OPENAI_API_KEY environment variable.")
                        st.stop()
                    agent = DataAgent(openai_api_key=api_key)
                    run_news_pipeline(agent, st.session_state.sources, st.container(), limit=10)

        with c2:
            actions_type = "primary" if st.session_state.active_tool == "Actions" else "secondary"
            if st.button("⚡ Actions", key="btn_actions", width='stretch', type=actions_type):
                st.session_state.active_tool = "Actions"
                output_area.empty()  # Clean Container

                with output_area:
                    st.write("Generating strategies...")

                    # Build control panel parameter dict - trade_price will be recommended by AI
                    trade_price = st.session_state.latest_price  # Use current market price as reference
                    position_oz = st.session_state.position_size_oz

                    # Calculate target profit in dollar amount: position_oz * price * profit_percent
                    # e.g., 1 oz at $2000 with 5% target = $100 profit → $2100 target price
                    if position_oz and trade_price:
                        target_profit_dollars = position_oz * trade_price * (st.session_state.target_profit / 100.0)
                        target_price = trade_price + target_profit_dollars
                    else:
                        target_price = trade_price  # Fallback to current price

                    context = {
                        "strategy": st.session_state.strategy,
                        "investment_level": st.session_state.investment,
                        "trade_mode": st.session_state.trade_mode,
                        "trade_price": trade_price,
                        "target_profit": target_price,  # Target price in USD (calculated)
                        "target_profit_pct": st.session_state.target_profit,
                        "risk_percent": st.session_state.risk_percent / 100.0,
                        "position_size_oz": position_oz,
                        "account_balance": st.session_state.account_balance,
                        "latest_price": st.session_state.latest_price,
                    }

                    # Resolve API key with fallback chain
                    api_key = _resolve_api_key(_get_user_api_key())
                    if not api_key:
                        st.error("No API key available. Please enter your OpenAI API key above, or set OPENAI_API_KEY environment variable.")
                        st.stop()

                    # Initialize TradingAgent correctly
                    t_agent = TradingAgent(
                        name="GoldTrader",
                        api_key=api_key,
                        json_path="gold_news.json",
                        context=context
                    )

                    # Call run() to generate decision with error handling
                    try:
                        decision = t_agent.run()
                    except Exception as e:
                        st.error(f"Error generating decision: {e}")
                        decision = None

                    if not decision:
                        st.markdown("### Actions\n\nNo decision produced.")
                        st.info("Please run **News** first to collect and analyze news data.")
                    else:
                        # Calculate position size if not manually set
                        trade_price = st.session_state.latest_price
                        pos_size = st.session_state.position_size_oz
                        if pos_size is None or pos_size <= 0:
                            confidence = decision.get("recommendation", {}).get("confidence", 5)
                            pos_size = t_agent.calculate_position_size(
                                account_balance=st.session_state.account_balance,
                                confidence=confidence,
                                user_risk_percent=st.session_state.risk_percent / 100.0,
                                gold_price=trade_price
                            )

                        # Tool function
                        def stars(n):
                            try:
                                n = int(n)
                            except Exception:
                                n = 0
                            return "★" * n + "☆" * (5 - n)


                        def action_text(a):
                            a = (a or "").upper()
                            if a == "BUY":  return "BUY"
                            if a == "SELL": return "SELL"
                            return "HOLD"


                        ms = decision.get("market_summary", {})
                        rec = decision.get("recommendation", {})
                        strategies = decision.get("strategies", [])

                        # Market Overview
                        st.caption(
                            f"Sentiment: {ms.get('sentiment', '—')} | "
                            f"Trend: {ms.get('trend', '—')} | "
                            f"Volatility: {ms.get('volatility', '—')}"
                        )

                        st.subheader("Strategy Options")

                        # Recommendation put first
                        best_id = rec.get("strategy_id")
                        ordered = []
                        if best_id:
                            best = next((s for s in strategies if s.get("id") == best_id), None)
                            if best:
                                ordered.append(best)
                        ordered += [s for s in strategies if s.get("id") != best_id]

                        # Three Cards
                        for idx, s in enumerate(ordered):
                            with st.container(border=True):
                                if idx == 0:
                                    st.success("✓ Recommended")

                                # Title
                                st.markdown(f"### [{s.get('id', '')}] {s.get('name', '—')}")

                                # Information
                                cols = st.columns([2, 3, 3])
                                with cols[0]:
                                    action = action_text(s.get('action'))
                                    if action == "BUY":
                                        st.markdown(
                                            '<span style="background-color: #28a745; color: white; padding: 4px 12px; border-radius: 15px; font-size: 14px;">📈 BUY</span>',
                                            unsafe_allow_html=True)
                                    elif action == "SELL":
                                        st.markdown(
                                            '<span style="background-color: #dc3545; color: white; padding: 4px 12px; border-radius: 15px; font-size: 14px;">📉 SELL</span>',
                                            unsafe_allow_html=True)
                                    else:
                                        st.markdown(
                                            '<span style="background-color: #6c757d; color: white; padding: 4px 12px; border-radius: 15px; font-size: 14px;">⏸ HOLD</span>',
                                            unsafe_allow_html=True)

                                with cols[1]:
                                    st.markdown(f"**Confidence:** {stars(s.get('confidence', 0))}")

                                with cols[2]:
                                    st.markdown(
                                        f"**Risk/Return:** {s.get('expected_risk', '—')}/{s.get('expected_return', '—')}")

                                # Display entry price and target price
                                entry_price = s.get('entry_price')
                                target_price = s.get('target_price')
                                if entry_price:
                                    st.markdown(f"**Entry:** ${entry_price:,.2f}")
                                if target_price:
                                    st.markdown(f"**Target:** ${target_price:,.2f}")

                                with st.expander("Rationale", expanded=(idx == 0)):
                                    st.write(s.get("rationale", ""))

                        # Display position size information
                        st.markdown("---")
                        st.subheader("Position Sizing")
                        st.info(
                            f"Risk: {st.session_state.risk_percent}% | "
                            f"Position: {pos_size:.2f} oz | "
                            f"Account: ${st.session_state.account_balance:,.0f}"
                        )


        # Risk button section
        with c3:
            risk_type = "primary" if st.session_state.active_tool == "Risk" else "secondary"
            if st.button("🛡️ Risk", key="btn_risk", width='stretch', type=risk_type):
                st.session_state.active_tool = "Risk"
                output_area.empty()  # Clean Container

                with output_area:
                    # Build control panel parameter dict
                    # Read price thresholds from trading decision (recommended strategy)
                    decision_path = "data/trading_decision.json"
                    if os.path.exists(decision_path):
                        with open(decision_path, "r", encoding="utf-8") as f:
                            decision = json.load(f)
                        # Get recommended strategy's prices
                        rec = decision.get("recommendation", {})
                        rec_strategy = next((s for s in decision.get("strategies", []) if s.get("name") == rec.get("name")), {})
                        buy_price = rec_strategy.get("entry_price") or rec_strategy.get("target_price") or 0.0
                        sell_price = rec_strategy.get("entry_price") or rec_strategy.get("target_price") or 0.0
                    else:
                        buy_price = st.session_state.latest_price or 0.0
                        sell_price = st.session_state.latest_price or 0.0

                    context = {
                        "strategy": st.session_state.strategy,
                        "investment_level": st.session_state.investment,
                        "buy_price_threshold": buy_price,
                        "sell_price_threshold": sell_price,
                        "target_profit": st.session_state.target_profit,
                        "latest_price": st.session_state.latest_price,
                    }

                    # Initialize RiskAgent correctly
                    api_key = _resolve_api_key(_get_user_api_key())
                    if not api_key:
                        st.error("No API key available. Please enter your OpenAI API key above, or set OPENAI_API_KEY environment variable.")
                        st.stop()
                    r_agent = RiskAgent(
                        api_key=api_key,
                        context=context
                    )

                    # Run risk assessment
                    run_risk_pipeline(r_agent, output_area)

                    # Check if file exists before reading
                    risk_report_path = "data/risk_report.json"
                    if not os.path.exists(risk_report_path):
                        st.warning("Risk report not found. Please run analysis first.")
                    else:
                        with open(risk_report_path, "r", encoding="utf-8") as f:
                            report = json.load(f)

                        summary = report.get("summary", {})

                        st.subheader("Risk Overview")
                        st.info(
                            f"Portfolio risk: {summary.get('portfolio_risk', '—')}\n\n"
                            f"{summary.get('comment', '')}"
                        )

                        st.subheader("Per-Strategy Risk")
                        items = report.get("items", [])
                        if len(items) < 3:
                            st.warning(f"Only {len(items)} strategy(ies) assessed. Expected 3 strategies.")
                        for it in items:
                            with st.container(border=True):
                                # Title
                                st.markdown(f"### Strategy #{it.get('id', '?')}")

                                # State：Approval Label + Score + Risk Stars
                                cols = st.columns([2, 3, 3])
                                with cols[0]:
                                    approval = it.get('approval', '').lower()
                                    if approval == 'approved':
                                        st.markdown(
                                            '<span style="background-color: #28a745; color: white; padding: 4px 12px; border-radius: 15px; font-size: 14px;">✓ Approved</span>',
                                            unsafe_allow_html=True)
                                    elif approval == 'rejected':
                                        st.markdown(
                                            '<span style="background-color: #dc3545; color: white; padding: 4px 12px; border-radius: 15px; font-size: 14px;">✗ Rejected</span>',
                                            unsafe_allow_html=True)
                                    else:
                                        st.markdown(
                                            '<span style="background-color: #ffc107; color: black; padding: 4px 12px; border-radius: 15px; font-size: 14px;">⊙ Conditional</span>',
                                            unsafe_allow_html=True)

                                with cols[1]:
                                    score = it.get('approval_score', 0)
                                    st.markdown(f"**Score:** {score:.1f}/10")

                                with cols[2]:
                                    risk_level = it.get('risk_level', '').lower()
                                    risk_map = {'low': 2, 'medium': 3, 'high': 4, 'very high': 5}
                                    risk_stars = risk_map.get(risk_level, 3)
                                    filled = "★" * risk_stars
                                    empty = "☆" * (5 - risk_stars)
                                    st.markdown(f"**Risk:** {filled}{empty}")

                                with st.expander("Key risks", expanded=False):
                                    for r in it.get("key_risks", []):
                                        st.write(f"- {r}")
                                with st.expander("Mitigations", expanded=False):
                                    for m in it.get("mitigations", []):
                                        st.write(f"- {m}")
                                if it.get("notes"):
                                    st.caption(it["notes"])

