# AIGoldAnalyst

> Powered and sponsored by [Goldtakas Technologies Limited](https://goldtakastechnologies.co.uk/)

An AI-powered virtual gold trading system that uses Large Language Models (LLMs) for market analysis, trading decisions, and risk management.

## Project Overview

AIGoldAnalyst is an intelligent gold trading decision system that integrates multiple AI agents:

- **Data Agent**: Collects and analyzes gold market news from multiple sources
- **Trading Agent**: Generates multi-strategy analysis and trading decisions based on market news
- **Risk Agent**: Evaluates trading strategy risks and provides risk control recommendations
- **History Agent**: Analyzes historical trading records and performance metrics

The system provides a web dashboard for visual analysis and interactive operations, powered by an automated pipeline.

## Project Structure

```
goldanalyst/
├── agents/                     # AI agent modules
│   ├── data_agent.py           # Data collection and news analysis
│   ├── trading_agent.py        # Trading decision generation
│   ├── risk_agent.py          # Risk assessment
│   ├── history_agent.py       # Historical data analysis
│   └── models/                # LLM model wrappers
│       ├── deepseek_agent.py
│       ├── claude_haiku_agent.py
│       ├── gpt4o_agent.py
│       └── gpt4o_ft_agent.py
├── dashboard/                 # Streamlit web dashboard
│   ├── app.py               # Main application entry
│   ├── auto_scheduler.py    # Automation scheduler
│   ├── auto_archive.py      # Data archiver
│   └── pipeline_runner.py   # Automated pipeline runner
├── config/                   # Configuration files
│   ├── trading.yaml
│   ├── llm.yaml
│   └── news_quality_scorer.yaml
├── test/                     # Test files
│   ├── test_data_agent.py
│   ├── test_trading_agent.py
│   ├── test_risk_agent.py
│   └── test_history_agent.py
├── dataset/                  # Datasets and training scripts
│   └── dataset/            # Training data (JSONL)
├── utils/                   # Utility modules
│   ├── validation.py
│   ├── llm_wrapper.py
│   ├── error_handler.py
│   └── config_paths.py
├── requirements.txt         # Dependencies
├── README.md
└── LICENSE                 # Apache 2.0
```

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/cem-cookie/goldanalyst.git
cd goldanalyst
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Activate
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

**Option A - Enter via Dashboard (recommended)**

Launch the dashboard and enter your API key directly in the UI:

```bash
streamlit run dashboard/app.py
```

**Option B - Environment Variable**

Create a `.env` file:

```bash
OPENAI_API_KEY=your_api_key_here
```

### 5. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard opens at `http://localhost:8501`.

### 6. Run Tests

```bash
python -m pytest test/ -v
```

## Dashboard Features

### Gold Price Chart

- Interactive gold price chart (XAU/USD)
- Technical indicators: Moving Averages (MA20, MA50), Bollinger Bands
- Time range selector: 3 years, 1 year, 6 months, 3 months
- Real-time price fetching via yfinance

### Control Panel

- **AI Model & API Key** - Select LLM provider (ChatGPT/Claude) and enter API key securely
- **Strategy** - Trading style: Scalping, Swing, or Seasonal
- **Investment** - Risk profile: Passive, Active, or Aggressive
- **Trade Mode** - Direction: Buy or Sell
- **Risk %** - Risk per trade (0.1% to 5%)
- **Position Size** - Manual entry or auto-calculate based on risk %
- **Account Balance** - USD value for position sizing

### Automation Settings

- **Auto-refresh** - Toggle automated pipeline execution
- **Pause** - Temporarily pause pipeline
- **Refresh Interval** - Set frequency (15/30/60/120/240 min)
- **API Usage Meter** - Tracks daily API request usage
- **Refresh Now** - Manually trigger full pipeline immediately
- **Error Tracking** - Alerts after 3+ consecutive failures
- **Archive Info** - Shows compressed data archive size

### Pipeline Steps

1. **Fetch Gold Price** - Get current gold price via yfinance
2. **Collect & Analyze News** - Gather news from sources, filter relevant articles, fetch full text
3. **Generate Trading Decision** - Multi-strategy analysis with confidence scores
4. **Risk Analysis** - Evaluate risk level and provide recommendations

### Trading History

- View historical trades and performance metrics
- Summary statistics: total trades, win rate, profit/loss
- Generate mock trading data for testing
- Recent trades table with expandable view

### News Collection

- **Sources**: Yahoo, Investing.com, MetalsDaily
- LLM-powered relevance filtering
- Full text fetching for selected articles
- Quality scoring algorithm
- Market summary generation via LLM

## Configuration

### Config Files

| File | Purpose |
|------|---------|
| `config/trading.yaml` | Position sizing and risk parameters |
| `config/llm.yaml` | LLM timeout and retry settings |
| `config/news_quality_scorer.yaml` | News quality scoring weights |

### Position Sizing

Configurable in `config/trading.yaml`:
- `max_risk_percent` - Default 2%
- `min_trade_size_oz` - Minimum trade size (0.01 oz)
- `max_trade_size_oz` - Maximum trade size (100 oz)

### LLM Timeout & Retry

- **Timeout**: Default 15 seconds (configurable)
- **Retries**: Up to 3 attempts with exponential backoff

## Testing

```bash
# Run all tests
python -m pytest test/

# Run individual test files
python -m pytest test/test_data_agent.py -v
python -m pytest test/test_trading_agent.py -v
python -m pytest test/test_risk_agent.py -v
python -m pytest test/test_history_agent.py -v
```

## Dependencies

- **streamlit** - Web dashboard framework
- **openai** - OpenAI API client
- **anthropic** - Claude API client
- **pandas/numpy** - Data processing
- **yfinance** - Gold price data
- **beautifulsoup4/newspaper3k** - Web scraping
- **altair** - Interactive charts
- **cryptography** - API key encryption
- **python-dotenv** - Environment variable loading

## Architecture Notes

### Agent Design

Each agent is a self-contained class with:
- A public API method (e.g., `run()`, `generate_trade_decision()`)
- Private helper methods prefixed with `_`
- Constructor-based or session-state API key injection
- Error handling with graceful fallback

### Model Selection

- **GPT-4o** (default)
- **Claude** (via UI selector)
- **DeepSeek** (via model registry)

### Data Flow

```
gold_news.json → TradingAgent → trading_decision.json
                                  ↓
                            RiskAgent → risk_report.json
```

## Security

API keys entered via dashboard UI are:
1. Encrypted with per-session Fernet key
2. Stored only in memory (session state)
3. Never written to disk or logs
4. Security warning displayed for user-provided keys

## Troubleshooting

### Import Errors

Ensure you're running from the project root directory.

### API Key Errors

- Verify the key was entered in the current session
- Check `.env` file exists and key is set correctly
- Ensure your API key has sufficient quota

### Port Already in Use

```bash
streamlit run dashboard/app.py --server.port 8502
```

### Missing Data Files

- `gold_history.csv` - Fetched automatically via yfinance on first run
- `gold_news.json` - Generated by running News pipeline
- `trading_decision.json` - Generated by running Actions pipeline
- `trades_history.csv` - Can be generated from dashboard

## License

Apache License 2.0 - See LICENSE file.

---

*For questions or contributions, open an issue on GitHub.*
