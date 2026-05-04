# AIGoldAnalyst

An AI-powered virtual gold trading system that uses Large Language Models (LLM) for market analysis, trading decisions, and risk management.

## 📋 Project Overview

AIGoldAnalyst is an intelligent gold trading decision system that integrates multiple AI agents:

- 📰 **Data Agent (DataAgent)**: Collects and analyzes gold market news from multiple sources (Yahoo, Investing.com, MetalsDaily)
- 💹 **Trading Agent (TradingAgent)**: Generates multi-strategy analysis and trading decisions based on market news
- 🛡️ **Risk Agent (RiskAgent)**: Evaluates trading strategy risks and provides risk control recommendations
- 📊 **History Agent (HistoryAgent)**: Analyzes historical trading records and performance metrics

The system provides a web dashboard for visual analysis and interactive operations, powered by an automated 4-step pipeline.

## 🏗️ Project Structure

```
AIGoldAnalyst/
├── agents/                    # AI agent modules
│   ├── data_agent.py         # Data collection and news analysis agent
│   ├── trading_agent.py      # Trading decision agent (GPT-4o-mini)
│   ├── risk_agent.py         # Risk assessment agent
│   └── history_agent.py      # Historical data analysis agent
├── dashboard/                # Streamlit web dashboard
│   ├── app.py               # Main application entry
│   ├── style.css            # Custom styles
│   ├── auto_scheduler.py    # Automation scheduler with rate limiting & error tracking
│   ├── auto_archive.py      # Data archiver (ZIP compression)
│   └── pipeline_runner.py   # 4-step automated pipeline runner
├── services/            # Reserved for future API services (v2+)
├── data/                    # Data directory
│   ├── gold_history.csv     # Gold price historical data (yfinance)
│   ├── gold_news.json       # Collected and analyzed news
│   ├── market_analysis.json # LLM-generated market analysis
│   ├── trading_decision.json # Structured trading decision output
│   ├── trades_history.csv   # Trading history records
│   └── archive/             # Compressed quarterly archives
├── mock_trades.py           # Mock trading data generator
├── test/                    # Test files
│   ├── test_data_agent.py
│   ├── test_trading_agent.py
│   └── test_risk_agent.py
├── config/                  # Configuration files
│   └── news_quality_scorer.yaml
├── dataset/                 # Datasets and training scripts
└── requirements.txt         # Python dependencies
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/SchuLearn/VirtualTradingOfficer.git
cd VirtualTradingOfficer
```

### 2. Create and Activate Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Key

**Option A — Enter via Dashboard (recommended)**

Launch the dashboard and enter your OpenAI API key directly in the UI. The key is encrypted per-session using Fernet and is never stored to disk or environment variables.

```bash
streamlit run dashboard/app.py
```

**Option B — Environment Variable**

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

The dashboard also supports Anthropic API keys (Claude) via the UI selector.

### 5. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open at `http://localhost:8501`.

### 6. Run Tests

```bash
# Test individual agents
python test/test_data_agent.py
python test/test_trading_agent.py
python test/test_risk_agent.py

# Or with pytest
python -m pytest test/
```

## 📖 Dashboard Guide

### Control Panel

The right-hand control panel lets you configure:

- **AI Model & API Key Settings** — Select LLM provider (ChatGPT/Claude) and enter your API key securely
- **Strategy** — Trading style: Scalping, Swing, or Seasonal
- **Investment** — Risk profile: Passive, Active, or Aggressive
- **Buy/Sell Price Thresholds** — Price boundaries for trade execution
- **Target Profit** — Desired profit margin

### Automation Settings

- **Auto-refresh** — Toggle automated pipeline execution on/off
- **Pause** — Temporarily pause the pipeline
- **Refresh Interval** — Set pipeline frequency (15/30/60/120/240 min)
- **API Usage Meter** — Tracks daily API request usage
- **Refresh Now** — Manually trigger the full pipeline immediately

### News Panel (📰)

1. Select news sources (Yahoo, Investing.com, MetalsDaily)
2. Click **News** to collect and analyze the latest gold market news
3. The system filters relevant articles via LLM, fetches full text, scores quality, and generates a market summary

### Actions Panel (⚡)

1. Ensure news data is loaded (run News first)
2. Click **Actions** to generate multi-strategy trading decisions
3. Choose from Conservative, Balanced, or Aggressive strategies with confidence scores, risk/return ratings, and a recommended action (BUY/SELL/HOLD)

### Risk Panel (🛡️)

Evaluates the risk level of the current trading recommendation and provides risk mitigation advice.

### Trading History (📊)

View historical trades and performance metrics. If no history exists, generate mock data with one click.

## 🔧 Automation System

### Pipeline Runner

The `PipelineRunner` executes a full 4-step pipeline:

```
Step 1: Fetch Gold Price  →  Step 2: Collect & Analyze News
    →  Step 3: Generate Trading Decision  →  Step 4: Risk Analysis
```

Can be triggered manually (Refresh Now) or scheduled via `AutoScheduler`.

### Auto Scheduler

- **RateLimiter** — Tracks API usage (default 1000 requests/day)
- **ErrorTracker** — Alerts after 3+ consecutive failures
- **LLMFallbackChain** — Attempts OpenAI → Claude on failure
- **Scheduling** — Configurable intervals (15–240 min)

### Auto Archiver

- Compresses data files into quarterly ZIP archives
- Uses maximum compression (ZIP_DEFLATED level 9)
- Archives stored in `data/archive/` with naming like `gold_2025_Q1.zip`

## 🔐 Secure API Key Handling

API keys entered via the dashboard UI are:

1. Encrypted immediately with a per-session Fernet key
2. Stored only in `st.session_state` (memory only, never persisted)
3. Decrypted in-memory only when needed for LLM calls
4. Never written to disk, environment variables, or logs

## 🧪 Testing

```bash
# Run all tests
python -m pytest test/

# Run individual tests
python test/test_data_agent.py
python test/test_trading_agent.py
python test/test_risk_agent.py
python test/test_historical_news_collection.py
```

## 📦 Dependencies

- **streamlit** — Web dashboard framework
- **openai** — OpenAI API client
- **anthropic** — Claude API client
- **pandas/numpy** — Data processing
- **yfinance** — Gold price data retrieval
- **beautifulsoup4/newspaper3k** — Web scraping and news parsing
- **altair** — Interactive charts
- **cryptography** — Fernet encryption for API key handling
- **python-dotenv** — Environment variable loading

See `requirements.txt` for the complete list.

## 🛠️ Troubleshooting

### Import Error: `ModuleNotFoundError: No module named 'mock_trades'`

Ensure you're running from the project root directory. Check that `mock_trades.py` exists in the root.

### API Key Errors

- If using the UI: make sure the key was entered in the current session
- If using `.env`: verify the file exists and `OPENAI_API_KEY` is set correctly
- Check that your API key has sufficient quota remaining

### Port Already in Use

```bash
# Use a different port
streamlit run dashboard/app.py --server.port 8502
```

### Missing Data Files

- `data/gold_history.csv` is fetched automatically via yfinance on first run
- Run the News pipeline to generate `data/gold_news.json`
- Run the Actions pipeline to generate `data/trading_decision.json`
- Generate mock trades: click **Generate mock trading data** in the History panel

### News Collection Fails

- Check your network connection
- Some sources may require VPN or proxy
- Review console output for specific error messages

## 📝 Architecture Notes

### Agent Design

Each agent is a self-contained class with:
- A public API method (e.g., `run()`, `generate_trade_decision()`)
- Private helper methods prefixed with `_`
- Session-state or constructor-based API key injection
- Error handling with graceful fallback and logging

### LLM Model Selection

- **ChatGPT (GPT-4o-mini)** — default for all pipeline steps
- **Claude (Anthropic)** — selectable via UI for decision generation
- Model selection is stored in `st.session_state['selected_model']`

### JSON Data Flow

```
gold_news.json  →  TradingAgent.load_news()  →  analyze_market_strategies()
                                              →  build_structured_decision()
                                              →  trading_decision.json

market_analysis.json  →  RiskAgent  →  risk_report.json
```

## ⚙️ Configuration

### Config Files

The application uses YAML configuration files in the `config/` directory:

| File | Purpose |
|------|---------|
| `trading.yaml` | Position sizing and risk parameters |
| `llm.yaml` | LLM timeout and retry settings |
| `news_quality_scorer.yaml` | News quality scoring weights |

### Position Sizing

Position sizing is calculated based on account balance, risk tolerance, and confidence:

```
position_size = (account_balance × risk_percent) / gold_price × confidence
```

**Configurable parameters** (in `config/trading.yaml`):
- `max_risk_percent` — Default 2% (0.02)
- `min_trade_size_oz` — Minimum trade size in ounces (default: 0.01)
- `max_trade_size_oz` — Maximum trade size in ounces (default: 100)

**UI controls**:
- Risk % — Slider (0.1% to 5%)
- Position size — Manual override or auto-calculate
- Account balance — USD value for sizing

### LLM Timeout & Retry

LLM calls are protected by timeout and retry logic:

- **Timeout**: Default 15 seconds (configurable in `config/llm.yaml`)
- **Retries**: Up to 3 attempts with exponential backoff (1s → 2s → 4s)

If all retries fail, the system returns a safe fallback response (e.g., HOLD action).

## 🚀 Future Features (v2)

The following features are planned for future releases:

### Trading Simulation & Backtesting
- **Portfolio Simulation**: Automatically simulate trades based on recommended strategies
- **P&L Projection**: Show potential profit/loss scenarios before executing real trades
- **Risk Metrics**: Display max drawdown, Sharpe ratio, win rate projections
- **What-If Analysis**: Allow users to modify parameters and see projected outcomes

### Enhanced AI Models
- **Model Comparison**: Side-by-side comparison of different LLM recommendations
- **Custom Fine-tuning**: Support for user-trained models on historical data
- **Ensemble Methods**: Combine multiple AI agents for more robust decisions

### Advanced Features
- **Multi-Asset Support**: Extend beyond gold to other precious metals (silver, platinum)
- **Technical Indicators Integration**: Add MA, RSI, MACD, Bollinger Bands analysis
- **News Sentiment Trends**: Track sentiment over time with historical charts
- **Alert System**: Push notifications for significant market events

### API Services
- **REST API Endpoints**: Expose trading signals via HTTP API
- **Webhook Integration**: Connect to broker APIs for automated execution

---

*Want to contribute or suggest features? Open an issue on GitHub!*
