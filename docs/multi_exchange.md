# Multi-Exchange Support

The bot now supports fetching data from multiple cryptocurrency exchanges, not just Bybit. This allows users to specify which exchange they want to get data from.

## Supported Exchanges

1. Bybit (Default)
2. Binance
3. KuCoin
4. Indodax
5. MEXC
6. OKX
7. Bitfinex
8. Gate.io
9. Kraken
10. Huobi

## Features

- **Exchange Detection**: Bot automatically detects which exchange the user wants to query based on their message.
- **Symbol Normalization**: Handles different symbol formats for different exchanges (e.g., "BTC-USDT" for KuCoin vs "BTCUSDT" for Binance).
- **Default Exchange**: Falls back to a default exchange (Bybit) if no specific exchange is mentioned.
- **Configurable**: Exchange support can be enabled/disabled and the default exchange can be changed.

## Usage Examples

Users can specify the exchange in their queries:

- "What's the price of Bitcoin on Binance?"
- "Get ETH price on KuCoin"
- "Show BTC price on Indodax"
- "Bybit server time"
- "What's BTC trading at on OKX?"

If no exchange is specified, the bot will use the default exchange (Bybit):

- "What's the price of Bitcoin?"
- "Get ETH price"
- "Show BTC price"

## Current Limitations

- Only server time and ticker data are currently supported for non-Bybit exchanges
- Private endpoints (wallet balance, positions) are only available for Bybit with API keys

## Technical Implementation

The implementation uses an `ExchangeClient` class that handles API requests to different exchanges, with appropriate normalization of symbols and formatting of responses.

Configuration settings:
- `MULTI_EXCHANGE_ENABLED`: Enable/disable multi-exchange support
- `DEFAULT_EXCHANGE`: Set the default exchange
- `AVAILABLE_EXCHANGES`: List of available exchanges