# Bybit Trading Bot - AI Agent Instructions

## Architecture Overview

This is a **unified Bybit V5 trading bot** with Telegram interface, LLM integration, and MCP server capabilities. The architecture follows a **public-first design** with clear private endpoint gating.

### Core Components

- **`src/main.py`**: Single unified entrypoint with `EnhancedBybitBot` class - handle Telegram commands and natural language queries
- **`src/bybit_client.py`**: Bybit V5 REST client with authentication, simulation modes, and strict public/private endpoint separation
- **`src/llm.py`**: OpenAI-compatible client (default: Novita GPT-OSS) with error handling for balance issues
- **`src/config.py`**: Environment-based configuration with public_only mode toggle
- **`tools/mcp_server.py`**: Optional MCP server for external AI tool integration

### Key Architectural Decisions

**Public-First Design**: Bot defaults to `BYBIT_PUBLIC_ONLY=true` mode. Private endpoints require explicit configuration and show clear guidance when unavailable.

**LLM Intent Analysis**: Natural language queries go through `analyze_user_intent()` which uses comprehensive API documentation context to route to appropriate Bybit endpoints.

**Dual Authentication Modes**: 
- Public-only: Market data, server time, tickers (no API keys needed)
- Full mode: Wallet balance, positions, trading (requires `BYBIT_API_KEY`/`BYBIT_API_SECRET`)

## Essential Editing Patterns

### When Editing Existing Files (DO NOT CREATE NEW FILES)

**Main.py Pattern**: Always edit `src/main.py` directly. DO NOT create `enhanced_main.py`, `main_complete.py`, or similar variants.

**Single Source of Truth**: Update existing files rather than creating duplicates. The codebase follows this principle strictly.

### API Client Extension Pattern

```python
# In bybit_client.py - Add new endpoints following this pattern:
async def new_endpoint(self, category: str, symbol: str) -> Dict[str, Any]:
    if self.simulation_mode:
        return self._simulate_endpoint_data(symbol)
    
    if self._client is None:
        return {"retCode": 10002, "retMsg": "Client not initialized", "result": {}}
    
    # Implementation with proper error handling
```

### LLM Integration Pattern

```python
# In main.py - Extend intent analysis in analyze_user_intent():
system_prompt = f"""
{API_DOCS_CONTEXT}  # Always include full API context

# Add new intents following documented API structure
"""
```

## Critical Development Workflows

### Environment Setup
```bash
# Required environment variables
TELEGRAM_BOT_TOKEN=your_telegram_token
ZAI_API_KEY=your_llm_api_key
BYBIT_PUBLIC_ONLY=true  # Start here, flip to false for trading

# Optional for private endpoints
BYBIT_API_KEY=your_bybit_key
BYBIT_API_SECRET=your_bybit_secret
```

### Testing Approach
```bash
# Test public endpoints (default mode)
python test_public_api.py

# Test specific functionality
python scripts/debug_llm.py
python scripts/debug_specific_query.py

# Run unified bot
python -m src.main
```

### Adding New Bybit Endpoints

1. **Check API Documentation**: Reference `api-docs.txt` for exact endpoint specification
2. **Update Client**: Add method to `bybit_client.py` following authentication patterns
3. **Update Intent Router**: Extend `analyze_user_intent()` in `main.py` with new action
4. **Update Executor**: Add case to `execute_api_call()` with proper error handling

## Project-Specific Conventions

### Error Handling Pattern
```python
# Always provide clear public/private mode guidance
if not self.bybit_client.can_access_private_endpoints():
    return (
        "🔒 *Feature Restricted*\n\n"
        "Your request requires API credentials.\n\n"
        "_To enable:_\n"
        "1. Set valid API credentials in `.env`\n"
        "2. Set `BYBIT_PUBLIC_ONLY=false`\n"
        "3. Restart bot"
    )
```

### LLM Response Formatting
- **Always include API context**: Use `API_DOCS_CONTEXT` constant in system prompts
- **Bilingual support**: Support both English and Indonesian naturally
- **Structured JSON**: LLM responses use consistent JSON schema for intent parsing

### File Organization Rules
- **No duplicate mains**: Edit `src/main.py` directly, never create variants
- **Single README**: Update existing `README.md`, don't create multiple documentation files
- **Centralized config**: All settings flow through `src/config.py` from environment

## Integration Points

### Telegram ↔ LLM ↔ Bybit Flow
```
User Message → analyze_user_intent() → execute_api_call() → Bybit API → Formatted Response
```

### MCP Server Integration
```python
# MCP tools mirror main bot capabilities
# tools/mcp_server.py provides same endpoints as Telegram interface
# Shares bybit_client.py and config.py for consistency
```

### Multi-Exchange Support
The codebase includes `exchange_client.py` and comprehensive API documentation in `api-docs.txt` covering 15+ exchanges (Binance, KuCoin, MEXC, etc.) for future expansion.

## Key Implementation Notes

- **Simulation Mode**: `bybit_client.py` includes realistic simulation for testing without live API
- **Rate Limiting**: Built into API client with proper error handling
- **Authentication**: HMAC-SHA256 signing implemented for Bybit V5 specification
- **Error Recovery**: LLM client handles provider balance issues with specific error codes

Focus on extending existing patterns rather than creating new architectures. The codebase is designed for incremental enhancement while maintaining the unified public-first approach.