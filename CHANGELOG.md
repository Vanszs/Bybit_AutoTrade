# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-09-27

### Major Rewrite - Multi-Exchange MCP Integration

#### Added
- **🔌 MCP Server Integration**: Complete FastAPI-MCP framework implementation
- **🌐 Multi-Exchange Support**: 6+ major exchanges (Bybit, Binance, KuCoin, OKX, Huobi, MEXC, Indodax)
- **🤖 Dual Interface**: Unified Telegram Bot + MCP Server architecture
- **💱 Currency Conversion**: Built-in USD/IDR converter with `/convert` command
- **📊 Price Comparison**: Real-time comparison across all supported exchanges
- **🧠 Enhanced Natural Language**: Improved LLM integration with contextual responses
- **📚 API Documentation Context**: Contextual help in every API response
- **🔒 Smart Error Handling**: User-friendly error messages and fallbacks
- **🎯 Public/Private Mode**: Clear separation with guided activation

#### Changed
- **Complete Architecture Rewrite**: From custom MCP to FastAPI-MCP framework
- **Unified Entry Point**: Single `main.py` with integrated MCP client
- **Enhanced Configuration**: Comprehensive environment variable system
- **Improved Telegram Bot**: Better command handling and markdown support
- **Streamlined File Structure**: Cleaned up redundant files and documentation

#### Removed
- **Legacy Files**: Removed outdated test files and duplicate implementations
- **Redundant MCP Components**: Consolidated multiple MCP implementations into one
- **Unused Dependencies**: Cleaned up requirements and imports

#### Fixed
- **🐛 Gate.io API Issues**: Fixed currency pair format (BTCUSDT → BTC_USDT)
- **🐛 Telegram Markdown Parsing**: Added fallback for broken markdown entities
- **🐛 IDR Price Formatting**: Proper thousands separator for Indonesian Rupiah
- **🐛 Private Endpoint Access**: Clear messaging for restricted features
- **🐛 MCP Transport Issues**: Resolved HTTP transport configuration

#### Technical Improvements
- **FastAPI-MCP Framework**: Modern MCP protocol implementation
- **Async/Await Patterns**: Consistent asynchronous programming throughout
- **Error Recovery**: Comprehensive exception handling with user feedback
- **Rate Limiting**: Built-in API rate limiting for all exchanges
- **Symbol Format Handling**: Automatic conversion between exchange formats
- **Type Safety**: Enhanced type hints and validation

#### Documentation
- **📖 Complete README Rewrite**: Step-by-step installation and deployment guide
- **🔧 Configuration Guide**: Comprehensive environment variable reference
- **🐳 Docker Support**: Complete containerization documentation
- **🧪 Testing Guide**: Comprehensive testing and debugging instructions
- **🚨 Troubleshooting**: Common issues and solutions
- **🔒 Security Guide**: API key management and best practices

## [1.x.x] - Previous Versions

### Legacy Implementation
- Initial Bybit V5 API integration
- Basic Telegram bot functionality
- Custom MCP server implementation
- Single exchange support
- Manual configuration system

---

**Migration Guide**: See [README.md](README.md) for complete setup instructions for v2.0.0.