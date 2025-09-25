Bybit LLM Telegram Bot

This project provides a minimal Telegram bot that uses an OpenAI-compatible API (default: Novita GPT‑OSS) to generate replies. It includes Bybit V5 integration and an optional MCP server scaffold so an MCP-compatible client can call into the same capabilities.

Quick Start

- Python 3.11+
- Create and fill a `.env` file (see `.env.example`).
- Install dependencies: `pip install -r requirements.txt`
- Run the bot: `python -m src.main`

Environment Variables

- `TELEGRAM_BOT_TOKEN`: Your Telegram bot token.
- `ZAI_API_KEY`: Your LLM provider API key (default expects Novita).
- `ZAI_BASE_URL` (optional): Defaults to `https://api.novita.ai/openai`.
- `LLM_ROUTER_MODEL` (optional): Lighter model for intent routing (defaults to `LLM_MODEL`).
- `BYBIT_API_KEY` and `BYBIT_API_SECRET` (optional): For Bybit integration (production API only).
- `BYBIT_PUBLIC_ONLY`: Set to "true" to use only public endpoints (default: "true").
- `BOT_AUTH_REQUIRED`: Set to "true" to require authentication (default: "false").

Files

- `src/main.py`: Bot entrypoint and handlers.
- `src/config.py`: Loads configuration from environment.
- `src/llm.py`: OpenAI-compatible LLM client (defaults to Novita GPT‑OSS).
- `src/bybit_client.py`: Bybit V5 REST client (server time, tickers, wallet-balance, create order stub).
- `tools/mcp_server.py`: Optional MCP server scaffold (experimental).

Notes

- Secrets are read from `.env`. Avoid hardcoding credentials.
- Bybit client implements signed V5 HTTP requests. Extend methods as needed.
- MCP is optional; run it only if you have an MCP client to connect.
- By default, the bot runs in public_only mode (only public API endpoints are available).
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
