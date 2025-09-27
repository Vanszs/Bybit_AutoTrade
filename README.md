# Bybit Multi-Exchange Trading Bot

🚀 **Unified Trading Bot** with dual interface: **Telegram Bot** + **MCP Server** for AI integration.

## 🎯 Overview

A comprehensive cryptocurrency trading system that combines:

- **📱 Telegram Bot**: Mobile-first interface with natural language processing
- **🔌 MCP Server**: Model Context Protocol server for AI tools integration (Claude, etc.)
- **🌐 Multi-Exchange Support**: 6+ major cryptocurrency exchanges
- **🤖 LLM Integration**: Natural language understanding for trading commands
- **📊 Real-time Data**: Live market data and price comparisons

## ✨ Key Features

### 🤖 Dual Interface Architecture
- **Telegram Bot**: Chat-based trading interface
- **MCP Server**: HTTP API for external AI integration
- **Unified Backend**: Shared trading engine and data sources

### 🌐 Multi-Exchange Support
| Exchange | Status | Features |
|----------|--------|----------|
| **Bybit** | ✅ Primary | Full API (public + private endpoints) |
| **Binance** | ✅ Public | Market data, tickers, prices |
| **KuCoin** | ✅ Public | Market data, tickers, prices |
| **OKX** | ✅ Public | Market data, tickers, prices |
| **Huobi** | ✅ Public | Market data, tickers, prices |
| **MEXC** | ✅ Public | Market data, tickers, prices |
| **Indodax** | ✅ Public | IDR market data (Indonesia) |

### 🧠 Intelligent Features
- **Natural Language Processing**: Understand commands like "What's Bitcoin price?"
- **Real-time Price Comparison**: Compare prices across all supported exchanges
- **Currency Conversion**: Built-in USD/IDR converter with live rates
- **Smart Error Handling**: User-friendly error messages and fallbacks
- **API Documentation Context**: Contextual help in every response

## 🛠️ Installation & Setup

### Prerequisites

- **Python 3.11+** (recommended)
- **Telegram Bot Token** (from @BotFather)
- **API Keys** (optional, for private endpoints)

### Step 1: Clone Repository

```bash
git clone https://github.com/Vanszs/Bybit_AutoTrade.git
cd Bybit_AutoTrade
```

### Step 2: Python Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Environment Configuration

Create `.env` file:

```bash
cp .env.example .env
```

Edit `.env` file:

```env
# ===============================
# REQUIRED SETTINGS
# ===============================

# Telegram Bot (Required)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here

# LLM API (Required for natural language)
ZAI_API_KEY=your_llm_api_key_here
ZAI_BASE_URL=https://api.novita.ai/openai

# ===============================
# OPTIONAL SETTINGS
# ===============================

# Bybit API (Optional - for private endpoints)
BYBIT_API_KEY=your_bybit_api_key
BYBIT_API_SECRET=your_bybit_secret
BYBIT_PUBLIC_ONLY=true

# LLM Configuration
LLM_MODEL=gpt-3.5-turbo
LLM_ROUTER_MODEL=gpt-3.5-turbo
LLM_TEMPERATURE=0.7

# Bot Configuration
BOT_AUTH_REQUIRED=false
BYBIT_TESTNET=false
```

### Step 4: Get Required API Keys

#### 🤖 Telegram Bot Token

1. Open Telegram and search for `@BotFather`
2. Send `/newbot` command
3. Follow instructions to create your bot
4. Copy the token and add to `.env` file

#### 🧠 LLM API Key (Novita GPT-OSS)

1. Visit [Novita AI](https://novita.ai)
2. Create account and get API key
3. Add to `.env` file as `ZAI_API_KEY`

#### 🔐 Bybit API Keys (Optional)

1. Visit [Bybit API Management](https://www.bybit.com/app/user/api-management)
2. Create API key with required permissions:
   - **Read**: Account info, positions, orders
   - **Trade**: Place/cancel orders (if needed)
3. Add to `.env` file
4. Set `BYBIT_PUBLIC_ONLY=false` to enable private endpoints

### Step 5: Verify Installation

```bash
# Test basic imports
python -c "
import sys
sys.path.insert(0, 'src')
from main import UnifiedTradingBot
print('✅ Installation successful!')
"
```

## 🚀 Running the System

### Option 1: Telegram Bot Only

```bash
python -m src.main
```

This starts the Telegram bot with integrated MCP client.

### Option 2: MCP Server Only

```bash
python tools/mcp_server.py
```

This starts the MCP server on `http://localhost:8001`.

### Option 3: Both (Recommended)

```bash
# Terminal 1: Start MCP Server
python tools/mcp_server.py

# Terminal 2: Start Telegram Bot  
python -m src.main
```

## 📱 Using the Telegram Bot

### Basic Commands

```bash
/start              # Initialize bot and see features
/help               # Complete command guide
/status             # System status and health check

# Market Data Commands
/price BTCUSDT bybit       # Get BTC price from Bybit
/price ETHUSDT binance     # Get ETH price from Binance  
/compare BTCUSDT           # Compare BTC across all exchanges
/exchanges                 # List all supported exchanges

# Utility Commands
/convert 100 USD to IDR    # Currency converter
/convert 1500000 IDR to USD

# Private Commands (requires API keys)
/balance                   # Get wallet balance
```

### Natural Language Examples

The bot understands natural language in **Indonesian** and **English**:

```text
"Harga Bitcoin sekarang?"
"What's the current ETH price?"
"Compare BTC prices across exchanges"
"Show me SOL price on KuCoin"
"Berapa harga DOGE di Binance?"
"Convert 50 USD to IDR"
```

## 🔌 MCP Server Integration

### For Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "bybit-trading": {
      "command": "python",
      "args": ["/path/to/Bybit_AutoTrade/tools/mcp_server.py"],
      "env": {
        "BYBIT_PUBLIC_ONLY": "true"
      }
    }
  }
}
```

### Direct HTTP API Usage

```bash
# Get server info
curl http://localhost:8001/

# Get Bybit ticker
curl "http://localhost:8001/bybit/tickers?category=spot&symbol=BTCUSDT"

# Compare prices across exchanges  
curl "http://localhost:8001/exchanges/compare-prices?symbol=BTCUSDT&exchanges=bybit,binance,kucoin"

# Get API documentation
curl "http://localhost:8001/api-docs?section=market"

# View API documentation
open http://localhost:8001/docs
```

## 🏗️ Project Structure

```
Bybit_AutoTrade/
├── 📄 README.md                    # This file
├── 📄 requirements.txt             # Python dependencies
├── 📄 .env.example                 # Environment template
├── 📄 api-docs.txt                 # API documentation context
│
├── 📁 src/                         # Main application code
│   ├── 📄 main.py                  # Unified bot entry point
│   ├── 📄 config.py                # Configuration management
│   ├── 📄 bybit_client.py          # Bybit V5 API client
│   ├── 📄 exchange_client.py       # Multi-exchange API client
│   ├── 📄 mcp_telegram_client.py   # MCP integration for Telegram
│   ├── 📄 llm.py                   # LLM client (natural language)
│   ├── 📄 auth.py                  # Authentication utilities
│   ├── 📄 features.py              # Technical analysis features
│   ├── 📄 model.py                 # ML models for prediction
│   ├── 📄 strategy.py              # Trading strategies
│   ├── 📄 natural_trading_assistant.py  # NLP trading assistant
│   └── 📄 xgb_model.py             # XGBoost model implementation
│
├── 📁 tools/                       # External tools
│   └── 📄 mcp_server.py            # FastAPI MCP server
│
├── 📁 scripts/                     # Utility scripts
│   ├── 📄 debug_llm.py             # LLM debugging
│   ├── 📄 debug_specific_query.py  # Query testing
│   ├── 📄 debug_tool_calls.py      # Tool call debugging
│   └── 📄 simple_llm_test.py       # Simple LLM test
│
├── 📁 docs/                        # Documentation
│   ├── 📄 MCP_TELEGRAM_INTEGRATION.md  # MCP integration guide
│   ├── 📄 multi_exchange.md        # Multi-exchange documentation
│   ├── 📄 natural_language_implementation.md  # NLP implementation
│   └── 📄 private_endpoints.md     # Private endpoint guide
│
└── 📁 data/                        # Runtime data
    └── 📄 auth.json                # Authentication store
```

## 🧪 Testing & Debugging

### Test Individual Components

```bash
# Test LLM integration
python scripts/debug_llm.py

# Test specific queries
python scripts/debug_specific_query.py

# Test tool calls
python scripts/debug_tool_calls.py

# Simple LLM functionality test
python scripts/simple_llm_test.py
```

### Test MCP Server

```bash
# Start MCP server
python tools/mcp_server.py

# Test endpoints (in another terminal)
curl http://localhost:8001/
curl "http://localhost:8001/bybit/tickers?category=spot&symbol=BTCUSDT"
curl "http://localhost:8001/exchanges/compare-prices?symbol=BTCUSDT"
```

### Test Exchange Connections

```bash
python -c "
import asyncio
import sys
sys.path.insert(0, 'src')

async def test_exchanges():
    from exchange_client import ExchangeClient
    client = ExchangeClient()
    
    exchanges = ['bybit', 'binance', 'kucoin']
    for exchange in exchanges:
        try:
            result = await client.get_ticker('BTCUSDT', exchange)
            print(f'✅ {exchange}: Connected')
        except Exception as e:
            print(f'❌ {exchange}: {str(e)}')
    
    await client.close()

asyncio.run(test_exchanges())
"
```

## 🔒 Security & Best Practices

### API Key Security

- **Never commit API keys** to version control
- **Use environment variables** for all sensitive data
- **Enable IP restrictions** on API keys when possible
- **Use read-only permissions** when trading is not required
- **Test with testnet first** before using mainnet

### Safe Development

```bash
# Always use testnet for development
BYBIT_TESTNET=true

# Start with public-only mode
BYBIT_PUBLIC_ONLY=true

# Enable private endpoints only when needed
BYBIT_PUBLIC_ONLY=false
```

### Production Deployment

```bash
# Disable debug logging
export LOG_LEVEL=INFO

# Use production API keys with proper restrictions
# Set up proper firewall rules
# Monitor API usage and rate limits
```

## 🐳 Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Create data directory
RUN mkdir -p data

# Expose MCP server port
EXPOSE 8001

# Default command (can be overridden)
CMD ["python", "-m", "src.main"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  bybit-bot:
    build: .
    environment:
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - ZAI_API_KEY=${ZAI_API_KEY}
      - BYBIT_PUBLIC_ONLY=true
    volumes:
      - ./data:/app/data
    restart: unless-stopped

  mcp-server:
    build: .
    command: python tools/mcp_server.py
    ports:
      - "8001:8001"
    environment:
      - BYBIT_PUBLIC_ONLY=true
    restart: unless-stopped
```

### Build and Run

```bash
# Build image
docker build -t bybit-trading-bot .

# Run Telegram bot
docker run -d --env-file .env bybit-trading-bot

# Run MCP server
docker run -d -p 8001:8001 --env-file .env bybit-trading-bot python tools/mcp_server.py
```

## 🔧 Configuration Guide

### Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | ✅ Yes | - | Telegram bot token from @BotFather |
| `ZAI_API_KEY` | ✅ Yes | - | LLM API key for natural language |
| `ZAI_BASE_URL` | No | novita.ai | LLM API base URL |
| `BYBIT_API_KEY` | No | - | Bybit API key (for private endpoints) |
| `BYBIT_API_SECRET` | No | - | Bybit API secret |
| `BYBIT_PUBLIC_ONLY` | No | `true` | Restrict to public endpoints only |
| `BYBIT_TESTNET` | No | `false` | Use Bybit testnet |
| `LLM_MODEL` | No | `gpt-3.5-turbo` | LLM model name |
| `LLM_TEMPERATURE` | No | `0.7` | LLM response creativity (0-1) |
| `BOT_AUTH_REQUIRED` | No | `false` | Require authentication for bot |

### Exchange Configuration

The bot supports multiple exchanges with different capabilities:

```python
# Public endpoints (no API key required)
exchanges = [
    'bybit',    # Primary exchange with full support
    'binance',  # Market data only
    'kucoin',   # Market data only  
    'okx',      # Market data only
    'huobi',    # Market data only
    'mexc',     # Market data only
    'indodax',  # Indonesian exchange (IDR pairs)
]

# Private endpoints (API key required)
private_exchanges = [
    'bybit',    # Trading, balance, positions
]
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Bot doesn't respond to commands

```bash
# Check bot token
echo $TELEGRAM_BOT_TOKEN

# Verify bot token with Telegram API
curl "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getMe"

# Check bot logs
python -m src.main
```

#### 2. LLM not working

```bash
# Test LLM connection
python scripts/simple_llm_test.py

# Check API key
echo $ZAI_API_KEY

# Verify API endpoint
curl -H "Authorization: Bearer $ZAI_API_KEY" $ZAI_BASE_URL/models
```

#### 3. Exchange API errors

```bash
# Test exchange connections
python -c "
import asyncio, sys
sys.path.insert(0, 'src')
from exchange_client import ExchangeClient

async def test():
    client = ExchangeClient()
    result = await client.get_ticker('BTCUSDT', 'bybit')
    print(result)
    await client.close()

asyncio.run(test())
"
```

#### 4. MCP Server not starting

```bash
# Check port availability
lsof -i :8001

# Start with debug logging
python tools/mcp_server.py --debug

# Test MCP endpoints
curl http://localhost:8001/
```

#### 5. Private endpoints not working

```bash
# Verify API keys are set
echo "API Key: ${BYBIT_API_KEY:0:8}..."
echo "Public only: $BYBIT_PUBLIC_ONLY"

# Test API key permissions
python -c "
import sys
sys.path.insert(0, 'src')
from bybit_client import BybitClient

client = BybitClient()
print('Can access private:', client.can_access_private_endpoints())
"
```

### Performance Optimization

```bash
# Monitor memory usage
python -c "
import psutil
print(f'Memory: {psutil.virtual_memory().percent}%')
print(f'CPU: {psutil.cpu_percent()}%')
"

# Check API rate limits
tail -f logs/api_requests.log

# Optimize for production
export PYTHONOPTIMIZE=1
export PYTHONDONTWRITEBYTECODE=1
```

## 📊 Monitoring & Logging

### Log Levels

```bash
# Debug (development)
export LOG_LEVEL=DEBUG

# Info (production)
export LOG_LEVEL=INFO

# Error only
export LOG_LEVEL=ERROR
```

### Health Checks

```bash
# Bot health check
curl http://localhost:8001/health

# Exchange connectivity
python scripts/debug_specific_query.py

# Database connectivity (if using)
python -c "import sqlite3; print('DB OK')"
```

## 🤝 Contributing

### Development Setup

```bash
# Clone and setup
git clone https://github.com/Vanszs/Bybit_AutoTrade.git
cd Bybit_AutoTrade

# Create development environment
python -m venv venv-dev
source venv-dev/bin/activate
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy

# Run tests
python -m pytest tests/

# Format code
black src/
flake8 src/
```

### Contribution Guidelines

1. **Fork** the repository
2. **Create feature branch**: `git checkout -b feature/new-feature`
3. **Make changes** with proper tests
4. **Follow code style**: Use black formatter
5. **Update documentation** if needed
6. **Test thoroughly** with both testnet and mainnet
7. **Submit pull request** with clear description

### Code Style

```bash
# Format Python code
black src/ tools/ scripts/

# Check style
flake8 src/ --max-line-length=100

# Type checking
mypy src/
```

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**Important**: This software is for educational and research purposes only.

- **Trading Risk**: Cryptocurrency trading involves substantial financial risk
- **No Warranty**: Software provided "as is" without warranty of any kind
- **Not Financial Advice**: This is not investment advice
- **User Responsibility**: Users are responsible for their own trading decisions
- **API Limits**: Respect exchange API rate limits and terms of service

### Risk Management

- **Start with testnet** and small amounts
- **Never invest more than you can afford to lose**
- **Understand the markets** before automated trading
- **Monitor bot behavior** closely
- **Have stop-loss strategies** in place

## 🆘 Support

### Documentation

- **📖 [Full API Documentation](docs/)** - Complete API reference
- **🔌 [MCP Integration Guide](docs/MCP_TELEGRAM_INTEGRATION.md)** - Claude integration
- **🌐 [Multi-Exchange Guide](docs/multi_exchange.md)** - Exchange-specific docs
- **🤖 [Natural Language](docs/natural_language_implementation.md)** - NLP features

### Community

- **🐛 [Issues](https://github.com/Vanszs/Bybit_AutoTrade/issues)** - Bug reports and feature requests
- **💬 [Discussions](https://github.com/Vanszs/Bybit_AutoTrade/discussions)** - Community discussions
- **📧 [Contact](mailto:your-email@example.com)** - Direct support

### Quick Links

- **📈 [Bybit API Docs](https://bybit-exchange.github.io/docs/v5/intro)** - Official Bybit API
- **🤖 [Telegram Bot API](https://core.telegram.org/bots/api)** - Telegram bot development
- **🔌 [Model Context Protocol](https://modelcontextprotocol.io/)** - MCP specification
- **🧠 [Novita AI](https://novita.ai/)** - LLM API provider

---

**🚀 Happy Trading with Multi-Exchange MCP Integration!**

*Built with ❤️ by the Bybit AutoTrade community*
- To enable trading and other private endpoints, set `BYBIT_PUBLIC_ONLY=false` and provide valid API keys.
- For faster small-talk replies, consider a lighter `LLM_ROUTER_MODEL` and smaller `LLM_MAX_TOKENS` (e.g., 512–2048).

Run MCP Server (optional)

- Install deps: `pip install -r requirements.txt`
- Start server: `python -m tools.mcp_server`
- The server exposes two example tools: `health_check` and `echo`. Extend to call `ZaiClient` or Bybit.

Telegram Commands

- `/start` and `/help`: Info bantuan.
- `/time`: Server time Bybit (public, production).
- `/ticker <category> [symbol]`: Ticker market V5, kategori `spot|linear|inverse|option`.
- `/balance [COIN]`: Wallet balance V5 (private, default `accountType=UNIFIED`).
- `/signal <category> <symbol> <interval> <mode>`: Prediksi XGBoost (scalping/swing) dari kline terkini.
- `/ask <pertanyaan>`: Natural language router → LLM memilih aksi (tickers, kline, orderbook, trades, balance, positions, orders, create/cancel order) dan bot mengeksekusi.
