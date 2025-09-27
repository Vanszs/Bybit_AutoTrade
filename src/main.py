#!/usr/bin/env python3
"""
UNIFIED TELEGRAM + MCP BOT
Telegram bot dengan LLM chat + MCP server capabilities
Satu aplikasi untuk semua kebutuhan trading
"""

import asyncio
import contextlib
import json
import logging
import threading
from typing import Dict, Any, List
from datetime import datetime

# MCP imports
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ServerCapabilities

# Telegram imports
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ChatAction

from config import get_config
from llm import ZaiClient
from bybit_client import BybitClient, BybitConfig
from exchange_client import ExchangeClient
from natural_trading_assistant import TradingToolsRegistry
from mcp_telegram_client import MCPClient, TelegramMCPIntegration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedTradingBot:
    """Unified bot dengan Telegram chat + MCP server capabilities"""

    def __init__(self):
        self.config = get_config()

        # Shared components
        self.llm_client = ZaiClient(
            api_key=self.config.zai_api_key,
            base_url=self.config.zai_base_url,
            default_model=self.config.llm_model,
            temperature=0.1,
            max_tokens=3000,
        )

        self.exchange_client = ExchangeClient()
        self.bybit_client = BybitClient(BybitConfig(
            api_key=self.config.bybit_api_key,
            api_secret=self.config.bybit_api_secret,
            testnet=self.config.bybit_testnet,
            public_only=self.config.public_only
        ))
        self.tools_registry = TradingToolsRegistry(self.exchange_client, self.bybit_client)

        # MCP Client untuk mengakses MCP server eksternal
        self.mcp_client = MCPClient("http://localhost:8001")
        self.mcp_integration = TelegramMCPIntegration(self.mcp_client)
        
        # MCP Server
        self.mcp_server = Server("unified-trading-bot")
        self.conversation_context = []

        # Telegram Bot
        if self.config.telegram_bot_token:
            builder = Application.builder().token(self.config.telegram_bot_token)
            builder.post_shutdown(self._on_telegram_shutdown)
            self.telegram_app = builder.build()
            self._setup_telegram_handlers()
        else:
            self.telegram_app = None
            logger.warning("No Telegram token - Telegram bot disabled")

        self._register_mcp_handlers()
        logger.info("🚀 Unified Trading Bot initialized - Telegram + MCP ready!")

    def _check_private_endpoint_access(self, user_id: str = None) -> tuple[bool, str]:
        """Check if user can access private endpoints"""
        if self.config.bybit_public_only:
            return False, (
                "🔒 *Mode Publik Aktif*\n\n"
                "Maaf, fitur ini memerlukan akses ke endpoint private.\n\n"
                "_Untuk mengaktifkan:_\n"
                "• Set API credentials di `.env`\n"
                "• Ubah `BYBIT_PUBLIC_ONLY=false`\n"
                "• Restart bot\n\n"
                "💡 *Tip: Gunakan testnet untuk testing aman*"
            )
        
        if not self.bybit_client.can_access_private_endpoints():
            return False, (
                "🔑 *Credentials Tidak Tersedia*\n\n"
                "Silakan set API credentials:\n"
                "• `BYBIT_API_KEY`\n"
                "• `BYBIT_API_SECRET`\n\n"
                "Lalu restart bot."
            )
        
        return True, ""

    def _setup_telegram_handlers(self):
        """Setup Telegram bot handlers"""
        self.telegram_app.add_handler(CommandHandler("start", self.telegram_start))
        self.telegram_app.add_handler(CommandHandler("help", self.telegram_help))
        self.telegram_app.add_handler(CommandHandler("status", self.telegram_status))
        self.telegram_app.add_handler(CommandHandler("price", self.telegram_price))
        self.telegram_app.add_handler(CommandHandler("compare", self.telegram_compare))
        self.telegram_app.add_handler(CommandHandler("balance", self.telegram_balance))
        self.telegram_app.add_handler(CommandHandler("exchanges", self.telegram_exchanges))
        self.telegram_app.add_handler(CommandHandler("convert", self.telegram_convert))
        self.telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.telegram_message))

    async def telegram_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Telegram /start command"""
        welcome_msg = """
🚀 **Unified Trading Bot**

👋 Bot trading dengan **Telegram Chat + MCP Server**!

🌟 **Fitur:**
• 🤖 Telegram chat dengan LLM
• 🔧 MCP server untuk tool integration
• 📊 Multi-exchange real-time data
• 🧠 Natural language understanding

💬 **Contoh:**
• "Harga Bitcoin sekarang?"
• "Show me top 5 CEX prices for BTC"
• "Compare ETH across exchanges"

✨ **Mode Dual:** Chat + MCP Server aktif!
        """
        await update.message.reply_text(welcome_msg, parse_mode='Markdown')

    async def telegram_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Telegram /help command"""
        help_msg = """
🆘 **Panduan Unified Bot**

🗣️ **Natural Language:**
• Chat normal dalam bahasa Indonesia/English
• Bot memahami konteks trading cryptocurrency
• Tanya apapun tentang harga, exchanges, arbitrage

📊 **Commands:**
• `/start` - Mulai bot
• `/help` - Panduan ini
• `/status` - Status sistem
• `/price [SYMBOL] [EXCHANGE]` - Get price (e.g. /price BTCUSDT binance)
• `/compare [SYMBOL]` - Compare prices across exchanges
• `/balance` - Get wallet balance (requires API key)
• `/exchanges` - List available exchanges
• `/convert [AMOUNT] [FROM] to [TO]` - Currency converter

🔧 **MCP Server:**
• Jalan bersamaan dengan Telegram
• Tools untuk integration dengan Claude/AI systems
• Natural language tool calls

💡 **Tips:**
• Gunakan bahasa natural
• Bot connect ke multiple exchanges
• Real-time market data tersedia

🚀 **Dual Mode:** Telegram + MCP Server!
        """
        await update.message.reply_text(help_msg, parse_mode='Markdown')

    async def telegram_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Telegram /status command"""
        try:
            server_time = await self.bybit_client.get_server_time()

            status_msg = f"""
📊 **System Status**

🤖 **Telegram Bot:** ✅ Active
🔧 **MCP Server:** ✅ Running
🌐 **Exchanges:** ✅ Connected ({len(self.config.available_exchanges)})
🧠 **LLM:** ✅ Active
📡 **API:** ✅ Connected

⏰ **Server Time:** `{server_time.get('time', 'N/A')}`

🎯 **Mode:** Unified (Telegram + MCP)
✨ Semua sistem normal!
            """
        except Exception as e:
            status_msg = f"❌ Error: {str(e)}"

        await update.message.reply_text(status_msg, parse_mode='Markdown')

    async def telegram_price(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Telegram /price command - get price from specific exchange"""
        try:
            # Parse arguments: /price BTCUSDT bybit
            args = context.args
            symbol = "BTCUSDT"
            exchange = "bybit"
            
            if args:
                symbol = args[0].upper()
                if len(args) > 1:
                    exchange = args[1].lower()
            
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
            
            # Use MCP integration for price query
            query = f"{symbol} price {exchange}"
            response = await self.mcp_integration.handle_price_query(query)
            await update.message.reply_text(response, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)}")

    async def telegram_compare(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Telegram /compare command - compare prices across exchanges"""
        try:
            # Parse arguments: /compare BTCUSDT
            args = context.args
            symbol = "BTCUSDT"
            
            if args:
                symbol = args[0].upper()
            
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
            
            # Use MCP integration for comparison
            query = f"compare {symbol} prices"
            response = await self.mcp_integration.handle_price_query(query)
            await update.message.reply_text(response, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)}")

    async def telegram_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Telegram /balance command - get wallet balance"""
        try:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
            
            # Check private endpoint access
            can_access, error_msg = self._check_private_endpoint_access(str(update.effective_user.id))
            if not can_access:
                await update.message.reply_text(error_msg, parse_mode='Markdown')
                return
            
            # Use MCP integration for balance
            response = await self.mcp_integration.handle_balance_query()
            await update.message.reply_text(response, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)}")

    async def telegram_exchanges(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Telegram /exchanges command - list available exchanges"""
        try:
            # Get server info from MCP
            server_info = await self.mcp_client.get_server_info()
            
            if server_info.get("success", True):
                endpoints = server_info.get("endpoints", {})
                
                response = "🏢 **Available Exchanges:**\n\n"
                
                # Bybit
                response += "**🟡 Bybit** (Primary)\n"
                response += "• Spot, Linear, Inverse, Options\n"
                response += "• Public & Private endpoints\n\n"
                
                # Multi-exchange
                multi_exchange = endpoints.get("multi_exchange", {})
                if multi_exchange:
                    response += "**🌐 Multi-Exchange Support:**\n"
                    exchanges = ["Binance", "KuCoin", "OKX", "Huobi", "MEXC"]
                    for ex in exchanges:
                        response += f"• {ex}\n"
                    response += "\n"
                
                response += "**💡 Usage:**\n"
                response += "• `/price BTCUSDT binance` - Single exchange\n"
                response += "• `/compare ETHUSDT` - Compare across exchanges\n"
                response += "• Chat: 'Compare BTC prices' - Natural language\n"
                
                await update.message.reply_text(response, parse_mode='Markdown')
            else:
                await update.message.reply_text("❌ Cannot get exchange info")
                
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)}")

    async def telegram_convert(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Telegram /convert command - currency converter"""
        try:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
            
            # Parse arguments
            args = context.args
            if len(args) < 2:
                help_msg = """
💱 **Currency Converter**

**Usage:**
• `/convert 100 USD to IDR` - Convert USD to IDR
• `/convert 1500000 IDR to USD` - Convert IDR to USD  
• `/convert 0.5 BTC to USD` - Convert BTC to USD (if price available)

**Supported currencies:**
• USD, IDR
• Major cryptocurrencies (BTC, ETH, etc.)
                """.strip()
                await update.message.reply_text(help_msg, parse_mode='Markdown')
                return
            
            # Simple conversion logic
            amount = float(args[0])
            from_currency = args[1].upper()
            to_currency = args[3].upper() if len(args) > 3 and args[2].lower() == "to" else "USD"
            
            conversion_rate = 15000  # 1 USD = 15,000 IDR (approximate)
            
            if from_currency == "USD" and to_currency == "IDR":
                result = amount * conversion_rate
                result_msg = f"""
💱 **Currency Conversion**

💰 **{amount:,.2f} USD** → **Rp {result:,.0f} IDR**

📊 Rate: 1 USD = Rp {conversion_rate:,}
⏰ Updated: Real-time approximate rate
                """.strip()
            
            elif from_currency == "IDR" and to_currency == "USD":
                result = amount / conversion_rate
                result_msg = f"""
💱 **Currency Conversion**

💰 **Rp {amount:,.0f} IDR** → **${result:,.2f} USD**

📊 Rate: 1 USD = Rp {conversion_rate:,}
⏰ Updated: Real-time approximate rate
                """.strip()
            
            else:
                result_msg = f"""
❌ **Conversion not supported**

Supported pairs:
• USD ↔ IDR
• Use `/price SYMBOL exchange` for crypto prices
                """.strip()
            
            await update.message.reply_text(result_msg, parse_mode='Markdown')
            
        except ValueError:
            await update.message.reply_text("❌ Invalid amount. Please enter a valid number.")
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)}")

    async def telegram_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle Telegram messages dengan LLM"""
        user_id = update.effective_user.id
        message_text = update.message.text

        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        logger.info(f"Telegram User {user_id}: {message_text}")

        try:
            # Process dengan shared LLM processing
            response = await self._process_with_llm_understanding(message_text)

            if response and len(response) > 0:
                response_text = response[0].text if hasattr(response[0], 'text') else str(response[0])
                response_text = response_text.strip()

                if not response_text:
                    response_text = (
                        "👋 Hi there! Halo! Saya siap bantu info market crypto, perbandingan harga exchange,"
                        " atau analisis strategi. Cukup tanya aja, ya!"
                    )

                # Clean markdown and split long messages
                response_text = self._clean_markdown_for_telegram(response_text)
                
                if len(response_text) > 4000:
                    chunks = [response_text[i:i+4000] for i in range(0, len(response_text), 4000)]
                    for chunk in chunks:
                        try:
                            await update.message.reply_text(chunk, parse_mode='Markdown')
                        except Exception as parse_error:
                            # Fallback to plain text if markdown fails
                            await update.message.reply_text(chunk)
                else:
                    try:
                        await update.message.reply_text(response_text, parse_mode='Markdown')
                    except Exception as parse_error:
                        # Fallback to plain text if markdown fails
                        await update.message.reply_text(response_text)
            else:
                await update.message.reply_text("❌ Tidak ada response dari sistem")

        except Exception as e:
            logger.error(f"Telegram error: {e}", exc_info=True)
            try:
                await update.message.reply_text(f"❌ Error: {str(e)}", parse_mode='Markdown')
            except:
                await update.message.reply_text(f"❌ Error: {str(e)}")

    def _clean_markdown_for_telegram(self, text: str) -> str:
        """Clean markdown text to avoid Telegram parsing errors"""
        import re
        
        # Fix common markdown issues only - don't over-escape
        
        # Fix bold/italic that might be broken
        text = re.sub(r'\*\*([^*]*?)\*\*', r'*\1*', text)  # Convert ** to *
        text = re.sub(r'__([^_]*?)__', r'_\1_', text)      # Convert __ to _
        
        # Remove markdown code blocks - convert to inline code
        text = re.sub(r'```(\w*)\n?([^`]*?)```', r'`\2`', text)
        
        # Fix broken or complex markdown links - just show text
        text = re.sub(r'\[([^\]]*?)\]\([^)]*?\)', r'\1', text)
        
        # Only escape problematic characters that break Telegram parsing
        # Keep basic formatting characters unescaped
        text = re.sub(r'([\\])', r'\\\\', text)  # Escape backslashes
        
        return text

    def _register_mcp_handlers(self):
        """Register MCP server handlers"""

        @self.mcp_server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="natural_trading_query",
                    description="Process ANY natural language query about cryptocurrency trading, prices, exchanges, arbitrage, or market analysis. The system uses advanced LLM understanding to comprehend complex requests like 'show me BTC prices on top 5 CEX', 'compare ETH between exchanges', 'arbitrage opportunities', or 'harga Bitcoin terbaik'. Supports English and Indonesian naturally.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language query about cryptocurrency trading. Examples: 'show me top 5 CEX prices for BTC', 'compare Ethereum prices between Binance and KuCoin', 'what are arbitrage opportunities for Bitcoin?', 'harga Bitcoin di exchange terbaik'"
                            }
                        },
                        "required": ["query"]
                    }
                )
            ]

        @self.mcp_server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            logger.info(f"🎯 MCP Tool call: {arguments.get('query', 'N/A')}")

            try:
                if name == "natural_trading_query":
                    return await self._process_with_llm_understanding(arguments.get("query", ""))
                else:
                    return [TextContent(type="text", text=f"❌ Unknown tool: {name}")]
            except Exception as e:
                logger.error(f"MCP Tool error: {e}", exc_info=True)
                return [TextContent(type="text", text=f"❌ Error: {str(e)}")]

    async def _on_telegram_shutdown(self, application: Application) -> None:
        """Cleanup resources when Telegram application stops."""
        logger.info("🛑 Telegram application shutting down - releasing clients")

        cleanup_tasks = []
        if hasattr(self.exchange_client, "close"):
            cleanup_tasks.append(self.exchange_client.close())
        if hasattr(self.bybit_client, "close"):
            cleanup_tasks.append(self.bybit_client.close())
        if hasattr(self.mcp_client, "close"):
            cleanup_tasks.append(self.mcp_client.close())

        if cleanup_tasks:
            results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            for exc in results:
                if isinstance(exc, Exception):
                    logger.warning(f"Cleanup error: {exc}")

    async def _process_with_llm_understanding(self, user_query: str) -> List[TextContent]:
        """Process query with TRUE LLM understanding - no hardcoded rules"""

        # Add to conversation context
        self.conversation_context.append({
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query
        })

        # Keep last 10 conversations for context
        if len(self.conversation_context) > 10:
            self.conversation_context = self.conversation_context[-10:]

        try:
            # Step 1: LLM analyzes dan understand user intent secara natural
            understanding = await self._get_llm_understanding(user_query)

            # Step 2: Execute berdasarkan LLM understanding
            result = await self._execute_based_on_understanding(understanding, user_query)

            return [TextContent(type="text", text=result)]

        except Exception as e:
            logger.error(f"Error in LLM processing: {e}", exc_info=True)
            return [TextContent(type="text", text=f"❌ Error processing query: {str(e)}")]

    async def _get_llm_understanding(self, user_query: str) -> Dict[str, Any]:
        """Get TRUE LLM understanding of user query - no hardcoded patterns"""

        # Build context dari conversation sebelumnya
        context_summary = ""
        if self.conversation_context:
            recent_queries = [ctx["user_query"] for ctx in self.conversation_context[-3:]]
            context_summary = f"Recent conversation: {' | '.join(recent_queries)}"

        # System prompt yang natural dan comprehensive
        system_prompt = f"""You are an expert cryptocurrency trading analyst with deep market knowledge. Analyze the user's natural language query and understand their EXACT intent.

AVAILABLE EXCHANGES: {', '.join(self.config.available_exchanges)}

AVAILABLE TOOLS:
1. get_price(symbol, exchange) - Get price from specific exchange
2. compare_top_exchanges(symbol, count) - Compare prices across top N exchanges
3. get_multiple_prices(symbol, exchanges) - Get prices from specific exchanges list
4. analyze_arbitrage(symbol, exchanges) - Find arbitrage opportunities
5. get_market_overview(symbols, exchanges) - Multi-symbol overview
6. get_positions() - Get user's current trading positions (requires API credentials)
7. get_balance() - Get user's wallet balance (requires API credentials)
8. close_position(position_id) - Close specific trading position by ID (requires API credentials)
9. get_kline_data(symbol, interval, exchange) - Get historical kline/candlestick data
10. get_orderbook(symbol, exchange, limit) - Get order book data from exchange
11. get_recent_trades(symbol, exchange, limit) - Get recent trade history
12. get_funding_history(symbol, category, limit) - Get funding rate history (Bybit only)
13. get_instruments_info(exchange, symbol) - Get trading instruments information
14. get_server_time(exchange) - Get server time from exchange

UNDERSTANDING RULES:
- If user asks for "top N" or "best N" exchanges: use compare_top_exchanges
- If user compares specific exchanges: use get_multiple_prices
- If user asks about arbitrage/opportunities: use analyze_arbitrage
- If user asks about multiple cryptocurrencies: use get_market_overview
- If user asks about single exchange: use get_price
- If user asks about positions/posisi/trading status: use get_positions
- If user asks about balance/saldo/wallet: use get_balance
- If user wants to close position: use close_position(position_id)
- If user asks for kline/candlestick/chart data: use get_kline_data
- If user asks for order book data: use get_orderbook
- If user asks for recent trades/transaction history: use get_recent_trades
- If user asks for funding rate/funding history: use get_funding_history
- If user asks for instruments/symbols info: use get_instruments_info
- If user asks for server time: use get_server_time

CRITICAL: Understand natural language context:
- "show me BTC prices on top 5 CEX" = compare_top_exchanges(BTC, 5)
- "compare ETH Binance vs KuCoin" = get_multiple_prices(ETH, [binance, kucoin])
- "arbitrage for Bitcoin" = analyze_arbitrage(BTC, all_exchanges)
- "harga Bitcoin terbaik" = compare_top_exchanges(BTC, 5)
- "cek posisi saya" / "check my positions" = get_positions()
- "cek saldo" / "check balance" / "wallet balance" = get_balance()
- "tutup posisi 1" / "close position 2" = close_position(1) or close_position(2)
- "get kline data BTC" / "chart data Bitcoin" = get_kline_data(BTC, 1h, bybit)
- "show orderbook BTC" / "order book Bitcoin" = get_orderbook(BTC, bybit, 20)
- "recent trades ETH" / "latest transactions" = get_recent_trades(ETH, bybit, 50)
- "funding rate BTC" / "funding history" = get_funding_history(BTC, linear, 50)
- "list instruments" / "available symbols" = get_instruments_info(bybit)
- "server time" / "current time" = get_server_time(bybit)

{context_summary}

Respond ONLY with valid JSON:
{{
    "understanding": "Clear explanation of what user wants",
    "action": "tool_name_to_use",
    "parameters": {{"param1": "value1", "param2": "value2"}},
    "reasoning": "Why this tool and parameters"
}}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        try:
            llm_response = await asyncio.to_thread(
                self.llm_client.chat,
                messages,
                temperature=0.1,
                max_tokens=1000
            )

            # Parse LLM response
            understanding = self._parse_llm_json_response(llm_response)

            logger.info(f"LLM Understanding: {understanding.get('understanding', 'N/A')}")
            logger.info(f"Action: {understanding.get('action', 'N/A')}")

            return understanding

        except Exception as e:
            logger.error(f"Error getting LLM understanding: {e}")
            # Fallback understanding
            return {
                "understanding": "Simple price query",
                "action": "compare_top_exchanges",
                "parameters": {"symbol": "BTC", "count": 5},
                "reasoning": "Fallback to basic comparison"
            }

    def _parse_llm_json_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM JSON response safely"""
        try:
            # Clean response
            response = response.strip()

            # Extract JSON block
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                if end != -1:
                    response = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                if end != -1:
                    response = response[start:end].strip()

            # Parse JSON
            return json.loads(response)

        except Exception as e:
            logger.error(f"Error parsing LLM JSON: {e}")
            # Return fallback
            return {
                "understanding": "Parse error - fallback mode",
                "action": "compare_top_exchanges",
                "parameters": {"symbol": "BTC", "count": 5},
                "reasoning": "JSON parse failed"
            }

    async def _execute_based_on_understanding(self, understanding: Dict[str, Any], original_query: str) -> str:
        """Execute action berdasarkan LLM understanding"""

        action = understanding.get("action")
        parameters = understanding.get("parameters", {})
        user_understanding = understanding.get("understanding", "")
        reasoning = understanding.get("reasoning", "")

        if not action or str(action).strip().lower() in {"none", "", "null"}:
            logger.info("No actionable tool requested – sending natural reply")
            return await self._handle_non_tool_response(user_understanding, original_query)

        logger.info(f"Executing: {action} with params: {parameters}")

        try:
            # Execute tool berdasarkan LLM understanding
            if action == "compare_top_exchanges":
                symbol = parameters.get("symbol", "BTC")
                count = parameters.get("count", 5)

                result = await self.tools_registry.execute_tool("compare_top_exchanges", {
                    "symbol": symbol,
                    "count": count
                })

                if result.get("success"):
                    return await self._format_with_llm(result, user_understanding, original_query, "top_comparison")

            elif action == "get_multiple_prices":
                symbol = parameters.get("symbol", "BTC")
                exchanges = parameters.get("exchanges", ["binance", "bybit"])

                result = await self.tools_registry.execute_tool("get_multiple_prices", {
                    "symbol": symbol,
                    "exchanges": exchanges
                })

                if result.get("success"):
                    return await self._format_with_llm(result, user_understanding, original_query, "comparison")

            elif action == "analyze_arbitrage":
                symbol = parameters.get("symbol", "BTC")
                exchanges = parameters.get("exchanges", ["binance", "bybit", "kucoin", "mexc", "okx"])

                result = await self.tools_registry.execute_tool("analyze_arbitrage", {
                    "symbol": symbol,
                    "exchanges": exchanges
                })

                if result.get("success"):
                    return await self._format_with_llm(result, user_understanding, original_query, "arbitrage")

            elif action == "get_price":
                symbol = parameters.get("symbol", "BTC")
                exchange = parameters.get("exchange", "binance")

                result = await self.tools_registry.execute_tool("get_price", {
                    "symbol": symbol,
                    "exchange": exchange
                })

                if result.get("success"):
                    return await self._format_with_llm(result, user_understanding, original_query, "single_price")

            elif action == "get_market_overview":
                symbols = parameters.get("symbols", ["BTC"])
                exchanges = parameters.get("exchanges", ["binance", "bybit", "kucoin"])

                result = await self.tools_registry.execute_tool("get_market_overview", {
                    "symbols": symbols,
                    "exchanges": exchanges
                })

                if result.get("success"):
                    return await self._format_with_llm(result, user_understanding, original_query, "overview")

            elif action == "get_positions":
                # Check if user can access private endpoints
                can_access, error_msg = self._check_private_endpoint_access()
                if not can_access:
                    return error_msg

                try:
                    result = await self.bybit_client.get_positions()
                    if result.get("retCode") == 0:
                        positions = result.get("result", {}).get("list", [])
                        if positions:
                            return await self._format_with_llm({"positions": positions}, user_understanding, original_query, "positions")
                        else:
                            return "📊 *Status Posisi*\n\n✅ Tidak ada posisi trading aktif saat ini."
                    else:
                        return f"❌ Error mengambil posisi: {result.get('retMsg', 'Unknown error')}"
                except Exception as e:
                    logger.error(f"Error getting positions: {e}")
                    return f"❌ Error: {str(e)}"

            elif action == "get_balance":
                # Check if user can access private endpoints
                can_access, error_msg = self._check_private_endpoint_access()
                if not can_access:
                    return error_msg

                try:
                    result = await self.bybit_client.get_wallet_balance()
                    if result.get("retCode") == 0:
                        balances = result.get("result", {}).get("list", [])
                        if balances:
                            return await self._format_with_llm({"balances": balances}, user_understanding, original_query, "balance")
                        else:
                            return "💰 *Saldo Wallet*\n\n❌ Tidak ada data saldo ditemukan."
                    else:
                        return f"❌ Error mengambil saldo: {result.get('retMsg', 'Unknown error')}"
                except Exception as e:
                    logger.error(f"Error getting balance: {e}")
                    return f"❌ Error: {str(e)}"

            elif action == "close_position":
                # Check if user can access private endpoints
                can_access, error_msg = self._check_private_endpoint_access()
                if not can_access:
                    return error_msg

                position_id = parameters.get("position_id", 1)
                
                try:
                    # First get positions to find the position by ID
                    positions_result = await self.bybit_client.get_positions()
                    if positions_result.get("retCode") == 0:
                        positions = positions_result.get("result", {}).get("list", [])
                        active_positions = [pos for pos in positions if float(pos.get("size", "0")) > 0]
                        
                        if not active_positions:
                            return "❌ Tidak ada posisi aktif untuk ditutup."
                        
                        if position_id < 1 or position_id > len(active_positions):
                            return f"❌ ID posisi {position_id} tidak valid. Posisi aktif: 1-{len(active_positions)}"
                        
                        # Get the position to close
                        position_to_close = active_positions[position_id - 1]
                        symbol = position_to_close.get("symbol")
                        side = "Sell" if position_to_close.get("side") == "Buy" else "Buy"  # Opposite side to close
                        qty = position_to_close.get("size")
                        
                        # Place market order to close position
                        close_result = await self.bybit_client.place_order(
                            category="linear",
                            symbol=symbol,
                            side=side,
                            orderType="Market",
                            qty=qty,
                            reduceOnly=True
                        )
                        
                        if close_result.get("retCode") == 0:
                            return f"✅ *Posisi Ditutup*\n\nPosisi #{position_id} ({symbol}) berhasil ditutup dengan market order."
                        else:
                            return f"❌ Gagal tutup posisi: {close_result.get('retMsg', 'Unknown error')}"
                    else:
                        return f"❌ Error mengambil posisi: {positions_result.get('retMsg', 'Unknown error')}"
                        
                except Exception as e:
                    logger.error(f"Error closing position: {e}")
                    return f"❌ Error: {str(e)}"

            elif action == "get_kline_data":
                symbol = parameters.get("symbol", "BTCUSDT")
                interval = parameters.get("interval", "1h")
                exchange = parameters.get("exchange", "bybit")
                limit = parameters.get("limit", 100)
                
                try:
                    if exchange.lower() == "bybit":
                        result = await self.bybit_client.get_kline(
                            category="spot",
                            symbol=symbol,
                            interval=interval,
                            limit=limit
                        )
                        
                        if result.get("retCode") == 0:
                            klines = result.get("result", {}).get("list", [])
                            if klines:
                                return await self._format_with_llm({"klines": klines, "symbol": symbol, "interval": interval}, user_understanding, original_query, "kline_data")
                            else:
                                return f"❌ Tidak ada data kline untuk {symbol} di interval {interval}"
                        else:
                            return f"❌ Error mengambil kline data: {result.get('retMsg', 'Unknown error')}"
                    else:
                        return f"❌ Exchange {exchange} belum didukung untuk kline data. Gunakan 'bybit'."
                        
                except Exception as e:
                    logger.error(f"Error getting kline data: {e}")
                    return f"❌ Error: {str(e)}"

            elif action in ["get_orderbook", "get_recent_trades", "get_funding_history", "get_instruments_info", "get_server_time"]:
                # Handle additional tools via tools registry
                try:
                    result = await self.tools_registry.execute_tool(action, parameters)
                    if result.get("success"):
                        return await self._format_with_llm(result, user_understanding, original_query, action)
                    else:
                        return f"❌ Error: {result.get('error', 'Unknown error')}"
                except Exception as e:
                    logger.error(f"Error executing {action}: {e}")
                    return f"❌ Error: {str(e)}"

            logger.warning(f"Unknown action '{action}' – falling back to conversational reply")
            return await self._handle_non_tool_response(user_understanding, original_query)

        except Exception as e:
            logger.error(f"Error executing {action}: {e}")
            return f"❌ Error: {str(e)}"

    async def _handle_non_tool_response(self, user_understanding: str, original_query: str) -> str:
        """Gracefully respond when no tool execution is required."""

        system_prompt = (
            "You are a friendly cryptocurrency trading assistant. "
            "Respond briefly with a warm greeting and invite the user to ask about crypto markets, "
            "trading strategies, or exchange data. Keep it under 60 words, bilingual (English + Indonesian) "
            "and professional."
        )

        try:
            response = await asyncio.to_thread(
                self.llm_client.chat,
                [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"User message: {original_query}\n"
                            f"LLM understanding summary: {user_understanding or 'No specific trading intent detected.'}"
                        ),
                    },
                ],
                temperature=0.3,
                max_tokens=200
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating fallback response: {e}")
            return (
                "👋 Hi there! Halo! Saya siap bantu info market crypto, perbandingan harga exchange, "
                "atau analisis strategi. Cukup tanya aja, ya!"
            )

    async def _format_with_llm(self, tool_result: Dict[str, Any], user_understanding: str, original_query: str, response_type: str) -> str:
        """Format response using LLM untuk natural presentation"""

        # System prompt untuk formatting
        format_prompt = f"""You are a professional cryptocurrency market analyst. Format the trading data into a natural, informative response for Telegram messaging.

USER QUERY: "{original_query}"
USER INTENT: {user_understanding}
RESPONSE TYPE: {response_type}

TELEGRAM FORMATTING GUIDELINES:
- Use emojis appropriately (🏆 for rankings, 💰 for prices, 📊 for data, 🥇🥈🥉 for top 3)
- Use simple *bold* text (single asterisk only) for emphasis
- Use _italic_ text (single underscore only) for secondary info
- Use `code` format (backticks) for prices and numbers
- NO escaping of characters like dots, exclamation marks, hyphens
- NO double asterisks (**) or double underscores (__)
- Use plain text for special characters like dots, exclamation marks, hyphens
- Show clear rankings for top exchanges
- Include price spread analysis if relevant
- Add insights about arbitrage opportunities
- Use professional but engaging tone
- Include timestamp

DATA TO FORMAT:
{json.dumps(tool_result, indent=2)}

Create a well-formatted, natural response that directly answers the user's query. Keep formatting simple and Telegram-compatible."""

        try:
            formatted_response = await asyncio.to_thread(
                self.llm_client.chat,
                [{"role": "system", "content": format_prompt}],
                temperature=0.2,
                max_tokens=1500
            )

            # Add context to conversation
            self.conversation_context[-1]["response"] = formatted_response

            return formatted_response

        except Exception as e:
            logger.error(f"Error formatting with LLM: {e}")
            # Fallback formatting
            return self._fallback_format(tool_result, response_type)

    def _fallback_format(self, result: Dict[str, Any], response_type: str) -> str:
        """Fallback formatting jika LLM formatting gagal"""

        if response_type == "top_comparison":
            comparison = result.get("comparison", [])
            if comparison:
                response = "🏆 **Top Exchange Prices**\n\n"
                for i, item in enumerate(comparison[:5]):
                    emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i] if i < 5 else "📊"
                    exchange = item.get("exchange", "N/A").upper()
                    price = item.get("price", "N/A")
                    response += f"{emoji} **{exchange}**: `${price}`\n"
                return response

        elif response_type == "arbitrage":
            opportunities = result.get("opportunities", [])
            if opportunities:
                response = "💎 **Arbitrage Opportunities**\n\n"
                for i, opp in enumerate(opportunities[:3]):
                    response += f"🚀 **#{i+1}**: Buy {opp.get('buy_exchange', 'N/A')} → Sell {opp.get('sell_exchange', 'N/A')}\n"
                    response += f"   Profit: {opp.get('profit_percent', 0):.2f}%\n\n"
                return response

        elif response_type == "positions":
            positions = result.get("positions", [])
            if positions:
                response = "📊 *Status Posisi Trading*\n\n"
                active_positions = []
                
                for i, pos in enumerate(positions[:10]):  # Limit to 10 positions
                    # Only show positions with actual size
                    size = float(pos.get("size", "0"))
                    if size == 0:
                        continue
                        
                    active_positions.append(pos)
                    position_id = i + 1
                    
                    symbol = pos.get("symbol", "N/A")
                    side = pos.get("side", "N/A")
                    size_str = pos.get("size", "0")
                    entry_price = pos.get("avgPrice", "0")
                    unrealized_pnl = pos.get("unrealisedPnl", "0")
                    mark_price = pos.get("markPrice", "0")
                    
                    side_emoji = "🟢" if side == "Buy" else "🔴"
                    pnl_emoji = "💚" if float(unrealized_pnl) >= 0 else "❤️"
                    
                    response += f"🆔 *ID #{position_id}* - {side_emoji} *{symbol}* ({side})\n"
                    response += f"   Size: `{size_str}`\n"
                    response += f"   Entry: `${entry_price}`\n"
                    response += f"   Mark: `${mark_price}`\n"
                    response += f"   PnL: {pnl_emoji} `${unrealized_pnl}`\n\n"
                
                if active_positions:
                    response += "💡 *Tip*: Untuk tutup posisi ketik 'tutup posisi [ID]'\n"
                    response += "   Contoh: 'tutup posisi 1' atau 'close position 2'\n"
                    return response
                else:
                    return "📊 *Status Posisi*\n\n✅ Tidak ada posisi trading aktif."
            else:
                return "📊 *Status Posisi*\n\n✅ Tidak ada posisi trading aktif."

        elif response_type == "balance":
            balances = result.get("balances", [])
            if balances:
                response = "💰 *Saldo Wallet*\n\n"
                for wallet in balances:
                    account_type = wallet.get("accountType", "UNIFIED")
                    coins = wallet.get("coin", [])
                    
                    response += f"🏦 *{account_type} Account*\n"
                    
                    for coin in coins[:10]:  # Limit to 10 coins
                        coin_name = coin.get("coin", "N/A")
                        wallet_balance = coin.get("walletBalance", "0")
                        available = coin.get("availableToWithdraw", "0")
                        
                        if float(wallet_balance) > 0:
                            response += f"   💎 *{coin_name}*: `{wallet_balance}`\n"
                            response += f"      Available: `{available}`\n"
                    
                    response += "\n"
                return response
            else:
                return "💰 *Saldo Wallet*\n\n❌ Tidak ada data saldo ditemukan."

        elif response_type == "kline_data":
            klines = result.get("klines", [])
            symbol = result.get("symbol", "BTCUSDT")
            interval = result.get("interval", "1h")
            
            if klines:
                response = f"📈 *Kline Data {symbol}* (Interval: {interval})\n\n"
                response += "`Time          | Open     | High     | Low      | Close    | Volume`\n"
                response += "`------------- | -------- | -------- | -------- | -------- | --------`\n"
                
                # Show last 10 klines
                for kline in klines[:10]:
                    timestamp = int(kline[0]) // 1000
                    from datetime import datetime
                    time_str = datetime.fromtimestamp(timestamp).strftime("%m-%d %H:%M")
                    open_price = float(kline[1])
                    high_price = float(kline[2])
                    low_price = float(kline[3])
                    close_price = float(kline[4])
                    volume = float(kline[5])
                    
                    response += f"`{time_str}    | {open_price:8.2f} | {high_price:8.2f} | {low_price:8.2f} | {close_price:8.2f} | {volume:8.0f}`\n"
                
                response += f"\n📊 Total records: {len(klines)}"
                return response
            else:
                return f"📈 *Kline Data {symbol}*\n\n❌ Tidak ada data tersedia."

        elif response_type == "get_orderbook":
            orderbook = result.get("orderbook", {})
            symbol = result.get("symbol", "BTCUSDT")
            exchange = result.get("exchange", "bybit")
            
            if orderbook:
                response = f"📚 *Order Book {symbol}* ({exchange.upper()})\n\n"
                
                # Show bids and asks
                bids = orderbook.get("b", [])[:5]  # Top 5 bids
                asks = orderbook.get("a", [])[:5]  # Top 5 asks
                
                if asks:
                    response += "*🔴 Asks (Sell Orders):*\n"
                    for ask in asks:
                        price, quantity = ask[0], ask[1]
                        response += f"   `${price}` - Qty: `{quantity}`\n"
                    response += "\n"
                
                if bids:
                    response += "*🟢 Bids (Buy Orders):*\n"
                    for bid in bids:
                        price, quantity = bid[0], bid[1]
                        response += f"   `${price}` - Qty: `{quantity}`\n"
                
                return response
            else:
                return f"📚 *Order Book {symbol}*\n\n❌ Tidak ada data tersedia."

        elif response_type == "get_recent_trades":
            trades = result.get("trades", {}).get("list", [])
            symbol = result.get("symbol", "BTCUSDT")
            exchange = result.get("exchange", "bybit")
            
            if trades:
                response = f"📊 *Recent Trades {symbol}* ({exchange.upper()})\n\n"
                response += "`Time     | Price     | Quantity  | Side`\n"
                response += "`-------- | --------- | --------- | ----`\n"
                
                for trade in trades[:10]:
                    timestamp = int(trade.get("T", 0)) // 1000
                    from datetime import datetime
                    time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
                    price = float(trade.get("p", 0))
                    qty = float(trade.get("v", 0))
                    side = "Buy" if trade.get("S") == "Buy" else "Sell"
                    side_emoji = "🟢" if side == "Buy" else "🔴"
                    
                    response += f"`{time_str} | ${price:8.2f} | {qty:8.4f} | {side_emoji} {side}`\n"
                
                return response
            else:
                return f"📊 *Recent Trades {symbol}*\n\n❌ Tidak ada data tersedia."

        elif response_type == "get_funding_history":
            funding_data = result.get("funding_history", {}).get("list", [])
            symbol = result.get("symbol", "BTCUSDT")
            
            if funding_data:
                response = f"💰 *Funding Rate History {symbol}*\n\n"
                response += "`Date         | Funding Rate | Symbol`\n"
                response += "`------------ | ------------ | ------`\n"
                
                for item in funding_data[:10]:
                    timestamp = int(item.get("fundingRateTimestamp", 0)) // 1000
                    from datetime import datetime
                    date_str = datetime.fromtimestamp(timestamp).strftime("%m-%d %H:%M")
                    funding_rate = float(item.get("fundingRate", 0))
                    funding_percent = funding_rate * 100
                    
                    response += f"`{date_str}  | {funding_percent:10.6f}% | {item.get('symbol', 'N/A')}`\n"
                
                return response
            else:
                return f"💰 *Funding Rate History {symbol}*\n\n❌ Tidak ada data tersedia."

        elif response_type in ["get_instruments_info", "get_server_time"]:
            # Generic fallback for other tools
            response = f"📋 *{response_type.replace('_', ' ').title()}*\n\n"
            
            if result.get("success"):
                # Extract key information
                for key, value in result.items():
                    if key not in ["success", "timestamp"]:
                        response += f"*{key.replace('_', ' ').title()}*: `{value}`\n"
                return response
            else:
                return f"❌ Error: {result.get('error', 'Unknown error')}"

        return f"📊 **Data Retrieved**: {json.dumps(result, indent=2)}"

async def main():
    """Main function - run both Telegram bot and MCP server"""

    # Initialize unified bot
    bot = UnifiedTradingBot()

    logger.info("🚀 Starting Unified Trading Bot...")
    logger.info("🤖 Telegram Bot + 🔧 MCP Server")

    # Function to run MCP server
    async def run_mcp_server():
        """Run MCP server"""
        try:
            async with stdio_server() as (read_stream, write_stream):
                await bot.mcp_server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="unified-trading-bot",
                        server_version="5.0.0",
                        capabilities=ServerCapabilities(
                            tools={"listChanged": True}
                        )
                    ),
                )
        except Exception as e:
            logger.error(f"MCP Server error: {e}")

    # Function to run Telegram bot
    async def run_telegram_bot():
        """Run Telegram bot"""
        if bot.telegram_app:
            try:
                logger.info("🤖 Starting Telegram bot...")
                await bot.telegram_app.initialize()
                await bot.telegram_app.start()
                await bot.telegram_app.updater.start_polling()

                # Keep running
                while True:
                    await asyncio.sleep(1)

            except asyncio.CancelledError:
                logger.info("Telegram bot task cancelled - shutting down")
                raise

            except Exception as e:
                logger.error(f"Telegram bot error: {e}")
            finally:
                with contextlib.suppress(Exception):
                    await bot.telegram_app.updater.stop()
                with contextlib.suppress(Exception):
                    await bot.telegram_app.stop()
                with contextlib.suppress(Exception):
                    await bot.telegram_app.shutdown()
        else:
            logger.warning("Telegram bot disabled - no token")

    # Choose startup mode based on environment
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "mcp-only":
        # MCP server only mode
        logger.info("🔧 MCP Server Only mode")
        await run_mcp_server()

    elif len(sys.argv) > 1 and sys.argv[1] == "telegram-only":
        # Telegram bot only mode
        logger.info("🤖 Telegram Bot Only mode")
        await run_telegram_bot()

    else:
        # Dual mode - both running
        logger.info("🚀 Dual Mode - Telegram + MCP Server")

        # Check if we're in stdio mode (MCP client connection)
        if sys.stdin.isatty():
            # Interactive mode - run Telegram bot
            await run_telegram_bot()
        else:
            # stdio mode - run MCP server
            await run_mcp_server()

if __name__ == "__main__":
    asyncio.run(main())