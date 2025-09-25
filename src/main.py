#!/usr/bin/env python3
"""
Enhanced Bybit Telegram Bot (Unified Main)
Bot yang memahami konteks API docs dan memberikan data real-time Bybit
Dengan respons yang jelas saat fitur private diminta pada mode publik.
"""

import asyncio
import json
import logging
import os
import sys
import re
import fcntl
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ChatAction

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ChatAction

# Import modules
from config import get_config
from llm import ZaiClient, LLMError  
from bybit_client import BybitClient, BybitConfig
from auth import AuthStore, AuthConfig

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# API Documentation context for LLM
API_DOCS_CONTEXT = """
# Bybit V5 API Documentation Summary

## Available Endpoints:
1. **Market Data** (Public):
   - /v5/market/time - Server time
   - /v5/market/tickers - Get ticker data for symbols
   - /v5/market/kline - Get kline/candlestick data
   - /v5/market/orderbook - Get order book
   - /v5/market/recent-trade - Get recent trades
   - /v5/market/instruments-info - Get instrument info

2. **Account** (Private):
   - /v5/account/wallet-balance - Get wallet balance
   - /v5/position/list - Get positions

3. **Categories**: spot, linear, inverse, option

## Response Format:
All responses return JSON with: retCode (0=success), retMsg, result, time

## Currency Pairs:
- If user only mentions currency like "BTC", assume pair with USDT (BTCUSDT)
- Common pairs: BTCUSDT, ETHUSDT, ADAUSDT, etc.
- Inverse pairs: BTCUSD, ETHUSD (no T at the end)
"""

class EnhancedBybitBot:
    """An enhanced Bybit Trading Bot with LLM integration and clear public/private handling."""
    
    def __init__(self):
        self.config = get_config()
        self.llm_client = ZaiClient(
            api_key=self.config.zai_api_key,
            base_url=self.config.zai_base_url,
            default_model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
        )
        self._router_model = self.config.llm_router_model or self.config.llm_model
        self.bybit_client = BybitClient(BybitConfig(
            api_key=self.config.bybit_api_key,
            api_secret=self.config.bybit_api_secret,
            testnet=self.config.bybit_testnet,
            demo=self.config.bybit_environment == "demo",
            simulation_mode=self.config.simulation_mode,
            simulation_balance=self.config.simulation_balance,
            public_only=self.config.public_only
        ))
        self.auth_store = AuthStore(AuthConfig(
            username=self.config.bot_auth_username or "admin",
            password=self.config.bot_auth_password or "admin123",
            store_path=self.config.bot_auth_store
        ))
        
        # Log the current mode
        logger.info(f"Bot started in {'public-only' if self.config.public_only else 'full'} mode")
        logger.info(f"Authentication {'required' if self.config.bot_auth_required else 'disabled'}")

    def normalize_symbol(self, user_input: str) -> str:
        """
        Normalize user input untuk symbol trading
        BTC -> BTCUSDT
        BTC/USD -> BTCUSD  
        ETH -> ETHUSDT
        """
        user_input = user_input.upper().replace("/", "").replace("-", "")
        
        # Jika sudah format lengkap (contoh: BTCUSDT, BTCUSD)
        if len(user_input) >= 6:
            return user_input
            
        # Jika hanya coin (BTC, ETH, dll), tambahkan USDT
        if len(user_input) <= 5:
            return f"{user_input}USDT"
            
        return user_input

    def is_probably_general_chat(self, text: str) -> bool:
        """Heuristic: detect small talk / non-trading to avoid extra LLM call."""
        t = text.lower()
        trading_keywords = [
            "btc", "eth", "price", "harga", "ticker", "kline", "candle",
            "orderbook", "order book", "volume", "wallet", "saldo", "balance",
            "position", "posisi", "leverage", "order", "buy", "sell",
            "spot", "linear", "inverse", "option", "perpetual", "funding",
            "symbol"
        ]
        return not any(k in t for k in trading_keywords)

    async def analyze_user_intent(self, message: str) -> Dict[str, Any]:
        """
        Gunakan LLM untuk menganalisis intent user dan menentukan API call yang diperlukan
        """
        system_prompt = f"""
        {API_DOCS_CONTEXT}
        
        Analyze the user message and determine:
        1. What Bybit API endpoint should be called?
        2. What parameters are needed?
        3. What category (spot/linear/inverse/option)?
        
        Return ONLY valid JSON with this structure:
        {{
            "action": "endpoint_name",
            "category": "spot|linear|inverse|option", 
            "symbol": "SYMBOL_IF_NEEDED",
            "params": {{"additional": "parameters"}},
            "explanation": "Brief explanation of what data will be fetched"
        }}
        
        Available actions: server_time, tickers, kline, orderbook, recent_trades, instruments_info, wallet_balance, positions
        
        If the message is not related to trading/market data, return:
        {{
            "action": "general_chat", 
            "explanation": "General conversation", 
            "reply": "Short, direct answer in <= 50 words"
        }}
        """
        
        try:
            response = await asyncio.to_thread(
                self.llm_client.chat,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                self._router_model,
                temperature=0.1,
                max_tokens=256,
            )
            
            # Parse JSON response
            intent = json.loads(response.strip())
            
            # Normalize symbol jika ada
            if intent.get("symbol"):
                intent["symbol"] = self.normalize_symbol(intent["symbol"])
                
            return intent
            
        except Exception as e:
            logger.error(f"Error analyzing intent: {e}")
            return {"action": "general_chat", "explanation": "Error analyzing request"}

    async def execute_api_call(self, intent: Dict[str, Any]) -> str:
        """
        Execute API call berdasarkan intent yang sudah dianalisis LLM
        """
        try:
            action = intent.get("action")
            category = intent.get("category", "spot")
            symbol = intent.get("symbol")
            params = intent.get("params", {})
            
            if action == "server_time":
                resp = await self.bybit_client.get_server_time()
                timestamp = resp.get('result', {}).get('timeSecond', 'N/A')
                human_time = datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H:%M:%S') if timestamp.isdigit() else 'N/A'
                return f"🕐 *Bybit Server Time*\n• Unix Timestamp: `{timestamp}`\n• Human Time: `{human_time}`"
                
            elif action == "tickers":
                resp = await self.bybit_client.get_tickers(category, symbol)
                return self.format_tickers_response(resp, symbol)
                
            elif action == "kline":
                limit = params.get("limit", 5)
                interval = params.get("interval", "1")
                resp = await self.bybit_client.get_kline(category, symbol, interval, limit)
                return self.format_kline_response(resp, symbol, interval)
                
            elif action == "orderbook":
                limit = params.get("limit", 10)
                resp = await self.bybit_client.get_orderbook(category, symbol, limit)
                return BybitClient.format_orderbook(resp)
                
            elif action == "recent_trades":
                limit = params.get("limit", 5)
                resp = await self.bybit_client.get_recent_trades(category, symbol, limit)
                return BybitClient.format_recent_trades(resp, limit)
                
            elif action == "instruments_info":
                resp = await self.bybit_client.get_instruments_info(category, symbol)
                return self.format_instruments_response(resp, symbol)
                
            elif action == "wallet_balance":
                if not self.bybit_client.can_access_private_endpoints():
                    mode = "public-only" if self.config.public_only and not self.config.simulation_mode else "restricted"
                    return (
                        "🔒 *Wallet Access Restricted*\n\n"
                        "Your request requires API credentials for private endpoint access.\n\n"
                        "*Current Configuration:*\n"
                        f"• Mode: `{mode}`\n"
                        "• Missing: `BYBIT_API_KEY` and `BYBIT_API_SECRET`\n\n"
                        "_To enable private endpoints:_\n"
                        "1. Set valid API credentials in your `.env` file\n"
                        "2. Set `BYBIT_PUBLIC_ONLY=false`\n"
                        "3. Restart the bot\n\n"
                        "_Note: You can still access all public market data._"
                    )
                resp = await self.bybit_client.get_wallet_balance()
                return BybitClient.format_wallet_balance(resp)
                
            elif action == "positions":
                if not self.bybit_client.can_access_private_endpoints():
                    mode = "public-only" if self.config.public_only and not self.config.simulation_mode else "restricted"
                    return (
                        "🔒 *Position Data Restricted*\n\n"
                        "Your request requires API credentials for private endpoint access.\n\n"
                        "*Current Configuration:*\n"
                        f"• Mode: `{mode}`\n"
                        "• Missing: `BYBIT_API_KEY` and `BYBIT_API_SECRET`\n\n"
                        "_To enable private endpoints:_\n"
                        "1. Set valid API credentials in your `.env` file\n"
                        "2. Set `BYBIT_PUBLIC_ONLY=false`\n"
                        "3. Restart the bot\n\n"
                        "_Note: You can still access all public market data._"
                    )
                resp = await self.bybit_client.get_position_list(category, symbol)
                return self.format_positions_response(resp)
                
            else:
                return "❌ API action tidak dikenali atau belum diimplementasi"
                
        except Exception as e:
            logger.error(f"Error executing API call: {e}")
            if "symbol not found" in str(e).lower():
                return f"❌ Symbol {symbol} tidak ditemukan di category {category}. Coba dengan category lain atau periksa ejaan symbol."
            return f"❌ Error mengambil data: {str(e)}"

    def format_tickers_response(self, resp: Dict[str, Any], symbol: Optional[str]) -> str:
        """Format response ticker menjadi user-friendly"""
        try:
            result = resp.get("result", {})
            list_items = result.get("list", [])
            
            if not list_items:
                return f"❌ Tidak ada data ticker untuk {symbol or 'symbol yang diminta'}"
                
            if len(list_items) == 1:
                # Single ticker with proper Markdown escaping
                ticker = list_items[0]
                return f"""📊 *{ticker.get('symbol', 'N/A')}*
💰 Price: `{ticker.get('lastPrice', 'N/A')}`
📈 24h Change: `{ticker.get('price24hPcnt', 'N/A')}%`
📊 Volume: `{ticker.get('volume24h', 'N/A')}`
🔺 High: `{ticker.get('highPrice24h', 'N/A')}`
🔻 Low: `{ticker.get('lowPrice24h', 'N/A')}`"""
            else:
                # Multiple tickers with proper Markdown
                lines = ["📊 *Market Tickers:*"]
                for ticker in list_items[:10]:  # Limit to 10
                    lines.append(f"• `{ticker.get('symbol', 'N/A')}`: `{ticker.get('lastPrice', 'N/A')}` (`{ticker.get('price24hPcnt', 'N/A')}%`)")
                
                if len(list_items) > 10:
                    lines.append(f"... dan {len(list_items) - 10} lainnya")
                    
                return "\n".join(lines)
                
        except Exception as e:
            return f"❌ Error memformat ticker: {str(e)}"

    def format_kline_response(self, resp: Dict[str, Any], symbol: str, interval: str) -> str:
        """Format response kline menjadi user-friendly"""
        try:
            result = resp.get("result", {})
            klines = result.get("list", [])
            
            if not klines:
                return f"❌ Tidak ada data kline untuk {symbol}"
                
            lines = [f"📈 **Kline {symbol} ({interval})**"]
            lines.append("```")
            lines.append("Time        | Open    | High    | Low     | Close   | Volume")
            lines.append("------------|---------|---------|---------|---------|--------")
            
            for kline in klines[:5]:  # Show last 5 candles
                timestamp = int(kline[0]) / 1000
                dt = datetime.fromtimestamp(timestamp)
                time_str = dt.strftime("%H:%M")
                
                lines.append(f"{time_str:11}| {kline[1]:7.2f} | {kline[2]:7.2f} | {kline[3]:7.2f} | {kline[4]:7.2f} | {float(kline[5]):8.0f}")
                
            lines.append("```")
            return "\n".join(lines)
            
        except Exception as e:
            return f"❌ Error memformat kline: {str(e)}"

    def format_instruments_response(self, resp: Dict[str, Any], symbol: Optional[str]) -> str:
        """Format response instruments info"""
        try:
            result = resp.get("result", {})
            list_items = result.get("list", [])
            
            if not list_items:
                return f"❌ Tidak ada info instrument untuk {symbol or 'yang diminta'}"
                
            instrument = list_items[0]
            return f"""ℹ️ **Instrument Info: {instrument.get('symbol', 'N/A')}**
📝 Status: {instrument.get('status', 'N/A')}
💱 Base Coin: {instrument.get('baseCoin', 'N/A')}
💰 Quote Coin: {instrument.get('quoteCoin', 'N/A')}
🎯 Min Order Qty: {instrument.get('lotSizeFilter', {}).get('minOrderQty', 'N/A')}
💲 Tick Size: {instrument.get('priceFilter', {}).get('tickSize', 'N/A')}"""
            
        except Exception as e:
            return f"❌ Error memformat instrument info: {str(e)}"

    def format_positions_response(self, resp: Dict[str, Any]) -> str:
        """Format response positions"""
        try:
            result = resp.get("result", {})
            list_items = result.get("list", [])
            
            if not list_items:
                return "📊 Tidak ada posisi terbuka"
                
            lines = ["📊 **Open Positions:**"]
            for pos in list_items:
                if float(pos.get("size", 0)) != 0:  # Only show non-zero positions
                    pnl = pos.get("unrealisedPnl", "0")
                    pnl_emoji = "📈" if float(pnl) >= 0 else "📉"
                    
                    lines.append(f"""• **{pos.get('symbol', 'N/A')}**
  Side: {pos.get('side', 'N/A')} | Size: {pos.get('size', 'N/A')}
  Entry: {pos.get('avgPrice', 'N/A')} | Mark: {pos.get('markPrice', 'N/A')}
  {pnl_emoji} PnL: {pnl}""")
                    
            return "\n".join(lines) if len(lines) > 1 else "📊 Tidak ada posisi terbuka"
            
        except Exception as e:
            return f"❌ Error memformat positions: {str(e)}"

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle semua pesan user dengan LLM integration"""
        # Auth check - only if authentication is required
        if self.config.bot_auth_required and not self.auth_store.is_authenticated(update.effective_user.id):
            await update.message.reply_text("🔐 Silakan login terlebih dahulu: /login <username> <password>")
            return

        user_message = update.message.text.strip()
        
        # Show typing indicator
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        
        # Fast path: if likely general chat, do single quick LLM call
        if self.is_probably_general_chat(user_message):
            try:
                response = await asyncio.to_thread(
                    self.llm_client.chat,
                    [
                        {"role": "system", "content": "You are Bybit_Bevan, a concise trading assistant using Novita GPT-OSS (OpenAI-compatible). Do not claim to be GPT-4. Keep answers short and factual."},
                        {"role": "user", "content": user_message}
                    ],
                    None,
                    temperature=0.5,
                    max_tokens=512,
                )
                await update.message.reply_text(response)
                return
            except Exception as e:
                try:
                    await update.message.reply_text(f"❌ Error LLM: {str(e)}")
                except Exception:
                    logger.exception("Failed to send Telegram message after LLM error")
                return

        # Analyze intent dengan LLM (router)
        intent = await self.analyze_user_intent(user_message)
        
        if intent.get("action") == "general_chat":
            # Use router-provided reply if present to avoid a second LLM call
            reply = intent.get("reply")
            if reply:
                try:
                    await update.message.reply_text(reply)
                    return
                except Exception:
                    logger.exception("Failed to send router reply; fallback to LLM call")
            try:
                response = await asyncio.to_thread(
                    self.llm_client.chat,
                    [
                        {"role": "system", "content": "You are Bybit_Bevan, a concise trading assistant using Novita GPT-OSS (OpenAI-compatible). Do not claim to be GPT-4. Keep answers short and factual."},
                        {"role": "user", "content": user_message}
                    ],
                    None,
                    temperature=0.5,
                    max_tokens=512,
                )
                await update.message.reply_text(response)
            except Exception as e:
                try:
                    await update.message.reply_text(f"❌ Error LLM: {str(e)}")
                except Exception:
                    logger.exception("Failed to send Telegram message after LLM fallback error")
        else:
            # Execute API call
            explanation = intent.get("explanation", "")
            if explanation:
                await update.message.reply_text(f"🔍 {explanation}")
                
            api_response = await self.execute_api_call(intent)
            try:
                await update.message.reply_text(api_response, parse_mode='MarkdownV2')
            except Exception as e:
                # Fallback to plain text if Markdown parsing fails
                logger.warning(f"Markdown parsing error: {e}")
                # Try with simpler Markdown
                try:
                    await update.message.reply_text(api_response, parse_mode='Markdown')
                except Exception:
                    # Last resort: plain text
                    await update.message.reply_text(api_response)

    async def start_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Start command handler"""
        await update.message.reply_text("""
🤖 **Bybit Trading Bot (Enhanced)**

Saya dapat membantu Anda:
• 📊 Mendapatkan data market real-time
• 💰 Cek harga cryptocurrency  
• 📈 Melihat grafik dan analisis
• 💼 Cek saldo dan posisi (perlu API key)

Contoh pertanyaan:
• "Harga BTC sekarang"
• "Tampilkan orderbook ETHUSDT"
• "Kline ADAUSDT 1 jam terakhir"
• "Cek saldo wallet saya"

Mulai dengan: /login <username> <password>
""")

    async def help_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Help command handler"""
        await update.message.reply_text("""
📖 **Bantuan Bot Trading**

Commands:
• /start - Mulai bot
• /help - Bantuan ini
• /login <user> <pass> - Login
• /logout - Logout

Cara bertanya:
Tanya langsung dengan bahasa natural, contoh:
• "Harga Bitcoin hari ini"
• "Volume trading ETH" 
• "Orderbook BNB spot"
• "Posisi saya sekarang"

Bot akan otomatis memahami dan mengambil data dari Bybit API yang sesuai! 🚀
""")

    async def login_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Login handler"""
        user_id = update.effective_user.id
        parts = update.message.text.split()
        
        if len(parts) < 3:
            await update.message.reply_text("Format: /login <username> <password>")
            return
            
        username, password = parts[1], parts[2]
        
        if self.auth_store.authenticate(user_id, username, password):
            await update.message.reply_text("✅ Login berhasil! Sekarang Anda bisa bertanya tentang data trading.")
        else:
            await update.message.reply_text("❌ Username atau password salah.")

    async def logout_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Logout handler"""
        self.auth_store.logout(update.effective_user.id)
        await update.message.reply_text("👋 Anda telah logout. Gunakan /login untuk masuk kembali.")

def main():
    """Main function"""
    bot = EnhancedBybitBot()
    
    # Create application
    app = Application.builder().token(bot.config.telegram_bot_token).build()

    # Add handlers
    app.add_handler(CommandHandler("start", bot.start_handler))
    app.add_handler(CommandHandler("help", bot.help_handler))
    app.add_handler(CommandHandler("login", bot.login_handler))
    app.add_handler(CommandHandler("logout", bot.logout_handler))
    
    # Handle all text messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))
    
    # Error handler to avoid crashes on network timeouts
    async def on_error(update, context: ContextTypes.DEFAULT_TYPE):
        logger.exception("Unhandled error in handler", exc_info=context.error)
        try:
            if update and getattr(update, 'message', None):
                await update.message.reply_text("⚠️ Timeout atau gangguan jaringan. Coba lagi sebentar.")
        except Exception:
            pass

    app.add_error_handler(on_error)
    
    # Start bot
    logger.info("🚀 Starting Enhanced Bybit Bot...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
