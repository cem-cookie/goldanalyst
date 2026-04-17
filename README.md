An AI-powered virtual gold trading system that uses Large Language Models (LLM) for market analysis, trading decisions, and risk management.

## 📋 Project Overview

Virtual Trading Officer is an intelligent gold trading decision system that integrates multiple AI agents:
- 📰 **Data Agent (DataAgent)**: Collects and analyzes gold market news from multiple sources
- 💹 **Trading Agent (TradingAgent)**: Generates trading strategies and decisions based on market analysis
- 🛡️ **Risk Agent (RiskAgent)**: Evaluates trading strategy risks and provides risk control recommendations
- 📊 **History Agent (HistoryAgent)**: Analyzes historical trading records and performance metrics

The system provides a web dashboard for visual analysis and interactive operations, along with RESTful API services for programmatic access.

## 🏗️ Project Structure

```
VirtualTradingOfficer/
├── agents/                    # AI agent modules
│   ├── data_agent.py         # Data collection and news analysis agent
│   ├── trading_agent.py      # Trading decision agent
│   ├── risk_agent.py         # Risk assessment agent
│   ├── history_agent.py      # Historical data analysis agent
│   └── models/               # Different LLM model implementations
│       ├── gpt4o_agent.py
│       ├── deepseek_agent.py
│       ├── claude_haiku_agent.py
│       └── ...
├── dashboard/                # Streamlit web dashboard
│   ├── app.py               # Main application entry
│   ├── components/          # UI components
│   └── style.css           # Style files
├── services/                # FastAPI backend services
│   ├── trading_service.py   # Trading service API
│   ├── data_service.py      # Data service API
│   └── risk_service.py      # Risk service API
├── data/                    # Data directory
│   ├── gold_history.csv    # Gold price historical data
│   ├── gold_news.json      # News data
│   ├── trades_history.csv  # Trading history
│   └── news/               # News data organized by date
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

Visit the project GitHub repository (https://github.com/SchuLearn/VirtualTradingOfficer) and clone the code locally:

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

### 4. Configure Environment Variables

Create a `.env` file (optional, if using environment variables to manage API keys):

```bash
# OpenAI API (for GPT models)
OPENAI_API_KEY=your_openai_api_key_here

# DeepSeek API (for DeepSeek models)
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Anthropic API (for Claude models)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

**Note**: If environment variables are not set, the system will attempt to read from environment variables. Some features may require at least one API key to be configured.

### 5. Launch Services

#### Launch Dashboard (Frontend)

```bash
streamlit run dashboard/app.py
```

The dashboard will automatically open in your browser at: `http://localhost:8501`

#### Start Backend Services (Optional)

If you need to use RESTful API services, you can start the backend services:

```bash
# Trading service (port 8000)
python -m uvicorn services.trading_service:app --reload --port 8000

# Data service (port 8081)
python -m uvicorn services.data_service:app --reload --port 8081

# Risk service (port 8083)
python -m uvicorn services.risk_service:app --reload --port 8083
```

**Note**: If running service files directly, default ports may differ (trading_service defaults to 8082). It's recommended to use the above commands to specify ports.

### 6. Run Tests

Run agent tests with:

```bash
# Test data agent
python test/test_data_agent.py

# Test trading agent
python test/test_trading_agent.py

# Test risk agent
python test/test_risk_agent.py
```

## 📖 User Guide

### Dashboard Features

1. **📰 News (News Analysis)**
   - Collect gold market news from multiple sources
   - Filter and analyze relevant news using LLM
   - Generate market summaries and trend analysis

2. **⚡ Actions (Trading Decisions)**
   - Generate trading strategies based on news analysis
   - Provide multiple strategy options and recommendations
   - Display confidence levels, risk/return ratios, and other information

3. **🛡️ Risk (Risk Assessment)**
   - Evaluate risk levels for each trading strategy
   - Provide risk mitigation recommendations
   - Display strategy approval status

4. **📊 Trading History**
   - View historical trading records
   - Analyze trading performance metrics
   - Visualize equity curves

### API Usage Examples

#### Trading Service

```bash
# Generate trading strategy
curl -X POST "http://localhost:8000/generate_strategy" \
  -H "Content-Type: application/json" \
  -d '{
    "json_path": "data/gold_news.json",
    "use_deepseek": true
  }'

# Check latest decision
curl "http://localhost:8000/check_latest"
```

#### Data Service

```bash
# Collect news
curl -X POST "http://localhost:8081/collect" \
  -H "Content-Type: application/json" \
  -d '{
    "sources": ["yahoo", "investing.com"],
    "limit": 10
  }'

# Analyze news
curl -X POST "http://localhost:8081/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "json_path": "data/gold_news.json",
    "min_quality_score": 30
  }'
```

## 🔧 Configuration

### News Quality Scoring Configuration

Edit `config/news_quality_scorer.yaml` to customize news quality scoring rules.

### Trading Parameters

In the dashboard control panel, you can adjust:
- **Strategy**: Trading strategy type (Scalping/Swing/Seasonal)
- **Investment**: Investment style (Passive/Active/Aggressive)
- **Buy/Sell Price Threshold**: Buy/sell price thresholds
- **Target Profit**: Target profit margin

## 🧪 Testing

The project includes multiple test files to verify component functionality:

```bash
# Run all tests
python -m pytest test/

# Run specific test
python test/test_data_agent.py
```

## 📦 Dependencies

Main dependencies:
- **streamlit**: Web dashboard framework
- **fastapi/uvicorn**: RESTful API services
- **openai**: OpenAI API client
- **anthropic**: Claude API client
- **pandas/numpy**: Data processing
- **yfinance**: Financial data retrieval
- **beautifulsoup4/newspaper3k**: Web scraping and news parsing
- **altair/matplotlib**: Data visualization

See `requirements.txt` for the complete list.

## 🛠️ Development

### Adding New LLM Models

1. Create a new agent class in the `agents/models/` directory
2. Inherit from or reference existing agent implementations
3. Integrate the new model in `agents/trading_agent.py`

### Adding New Data Sources

1. Add a new data source method in `agents/data_agent.py`
2. Update the news source selector in the dashboard

## 📝 Important Notes

1. **API Keys**: Ensure necessary API keys are set to use LLM features
2. **Data Files**: Initial data files may need to be downloaded or generated on first run
3. **Network Connection**: News collection features require network connectivity
4. **Python Version**: Python 3.9+ is recommended

## 🔍 Troubleshooting

### Common Issues

1. **Import Error: `ModuleNotFoundError: No module named 'mock_trades'`**
   - Ensure you're running commands from the project root directory
   - Check that `mock_trades.py` exists in the root directory

2. **API Key Errors**
   - Ensure correct environment variables are set
   - Check that `.env` file exists and is formatted correctly
   - Verify API keys are valid and have sufficient quota

3. **Port Already in Use**
   - Change the port number: `--port 8001`
   - Or stop the process using the port

4. **Missing Data Files**
   - Run `python dataset/scripts/mock_trades.py` to generate mock trading data
   - Ensure `data/gold_history.csv` exists (can be obtained from yfinance)

5. **Streamlit Won't Start**
   - Check if streamlit is installed: `pip install streamlit`
   - Ensure you're running the command from the correct directory

6. **News Collection Fails**
   - Check network connection
   - Some news sources may require VPN or proxy
   - Check console error messages

# Trading AI Application

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/trading_ai.git
   ```

2. Set environment variables:
   ```bash
   export EAS_TOKEN=your_secret_token
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python main.py
   ```
