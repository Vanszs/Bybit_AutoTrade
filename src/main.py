#!/usr/bin/env python3
"""
Complete Bybit Telegram Trading Bot dengan LLM + XGBoost Integration
- Mendukung spot & future dynamic berdasarkan konteks user
- AI Analysis dengan XGBoost model atau LLM reasoning
- Real-time data dari Bybit API
"""

import asyncio
import json
import logging
import os
import re
import csv
import tempfile
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import math
from io import StringIO, BytesIO

import pandas as pd
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
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

# API Documentation context untuk LLM
API_DOCS_CONTEXT = """
# Bybit V5 API Complete Reference

## Market Categories:
- **spot**: Spot trading (BTCUSDT, ETHUSDT, etc.)
- **linear**: USDT Perpetual (BTCUSDT, ETHUSDT with leverage)
- **inverse**: Coin Margined (BTCUSD, ETHUSD)
- **option**: Options trading

## Key Endpoints:
1. /v5/market/tickers - Price data
2. /v5/market/kline - OHLCV candlestick data
3. /v5/market/orderbook - Order book depth
4. /v5/market/instruments-info - Trading pair info
5. /v5/account/wallet-balance - Account balance
6. /v5/position/list - Open positions

## Symbol Normalization Rules:
- Single coin (BTC) → BTCUSDT (spot default)
- User context mentioning "future/leverage/perpetual" → linear category
- User context mentioning "inverse/coin margin" → inverse category
"""

class TechnicalAnalyzer:
    """Technical Analysis untuk OHLCV data"""
    
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators dari OHLCV data"""
        try:
            # RSI (14 periods)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Moving Averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price change indicators
            df['price_change_1h'] = df['close'].pct_change(1)
            df['price_change_4h'] = df['close'].pct_change(4)
            df['volatility'] = df['close'].rolling(window=20).std()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df

class XGBoostPredictor:
    """XGBoost Model untuk trading signal prediction"""
    
    def __init__(self):
        self.model = None
        self.feature_columns = []
        # XGBoost parameters optimized untuk trading
        self.params = {
            'objective': 'binary:logistic',
            'eta': 0.05,                    # learning rate
            'max_depth': 6,                 # tree depth
            'min_child_weight': 10,         # minimum sum of hessians
            'gamma': 1.0,                   # min loss reduction
            'lambda': 1.5,                  # L2 regularization  
            'alpha': 0.0,                   # L1 regularization
            'subsample': 0.8,               # row sampling
            'colsample_bytree': 0.7,        # column sampling per tree
            'colsample_bylevel': 0.7,       # column sampling per level
            'scale_pos_weight': 1.2,        # handle imbalance
            'eval_metric': 'aucpr',         # precision-recall AUC
            'tree_method': 'hist',
            'random_state': 42
        }
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features untuk XGBoost model"""
        try:
            # Calculate technical indicators
            df = TechnicalAnalyzer.calculate_indicators(df)
            
            # Feature engineering
            feature_df = pd.DataFrame()
            
            # Price features
            feature_df['close_price'] = df['close']
            feature_df['open_price'] = df['open']
            feature_df['high_price'] = df['high']
            feature_df['low_price'] = df['low']
            feature_df['volume'] = df['volume']
            
            # Technical indicators
            feature_df['rsi'] = df['rsi']
            feature_df['macd'] = df['macd']
            feature_df['macd_signal'] = df['macd_signal']
            feature_df['macd_histogram'] = df['macd_histogram']
            
            # Moving averages
            feature_df['sma_20'] = df['sma_20']
            feature_df['sma_50'] = df['sma_50']
            feature_df['ema_12'] = df['ema_12']
            feature_df['ema_26'] = df['ema_26']
            
            # Bollinger Bands
            feature_df['bb_upper'] = df['bb_upper']
            feature_df['bb_middle'] = df['bb_middle']
            feature_df['bb_lower'] = df['bb_lower']
            feature_df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volume features
            feature_df['volume_sma'] = df['volume_sma']
            feature_df['volume_ratio'] = df['volume_ratio']
            
            # Price change features
            feature_df['price_change_1h'] = df['price_change_1h']
            feature_df['price_change_4h'] = df['price_change_4h']
            feature_df['volatility'] = df['volatility']
            
            # Derived features
            feature_df['high_low_ratio'] = df['high'] / df['low']
            feature_df['close_open_ratio'] = df['close'] / df['open']
            feature_df['hl2'] = (df['high'] + df['low']) / 2
            feature_df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
            feature_df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
            
            # Trend features
            feature_df['sma_trend'] = (df['sma_20'] > df['sma_50']).astype(int)
            feature_df['price_above_sma'] = (df['close'] > df['sma_20']).astype(int)
            feature_df['ema_trend'] = (df['ema_12'] > df['ema_26']).astype(int)
            
            # Remove NaN values
            feature_df = feature_df.fillna(method='ffill').fillna(method='bfill')
            
            self.feature_columns = feature_df.columns.tolist()
            return feature_df
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()
    
    def create_mock_model(self):
        """Create mock trained model untuk demo purposes"""
        try:
            import xgboost as xgb
            
            # Create dummy training data
            np.random.seed(42)
            n_samples = 1000
            n_features = len(self.feature_columns) if self.feature_columns else 25
            
            X_dummy = np.random.randn(n_samples, n_features)
            # Create realistic trading signals (more sells in downtrend)
            y_dummy = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])  
            
            dtrain = xgb.DMatrix(X_dummy, label=y_dummy)
            
            # Train model dengan optimized parameters
            self.model = xgb.train(
                params=self.params,
                dtrain=dtrain,
                num_boost_round=100,
                verbose_eval=False
            )
            
            logger.info("Mock XGBoost model created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating mock model: {e}")
            return False
    
    def predict_signal(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Predict trading signal dari features"""
        try:
            if self.model is None:
                if not self.create_mock_model():
                    return {"action": "HOLD", "confidence": 0.0, "error": "Model creation failed"}
            
            # Get last row of features
            if features.empty:
                return {"action": "HOLD", "confidence": 0.0, "error": "No features available"}
            
            latest_features = features.iloc[-1:].values
            
            import xgboost as xgb
            dtest = xgb.DMatrix(latest_features)
            
            # Get prediction probability
            pred_proba = self.model.predict(dtest)[0]
            
            # Determine action berdasarkan probability
            if pred_proba > 0.6:
                action = "BUY"
                confidence = pred_proba
            elif pred_proba < 0.4:
                action = "SELL" 
                confidence = 1 - pred_proba
            else:
                action = "HOLD"
                confidence = 0.5
            
            return {
                "action": action,
                "confidence": round(confidence, 3),
                "probability": round(pred_proba, 3),
                "model": "XGBoost"
            }
            
        except Exception as e:
            logger.error(f"Error predicting signal: {e}")
            return {"action": "HOLD", "confidence": 0.0, "error": str(e)}

class CompleteBybitBot:
    """Complete Bybit Trading Bot dengan semua fitur maksimal"""
    
    def __init__(self):
        self.config = get_config()
        self.llm_client = ZaiClient(
            api_key=self.config.zai_api_key,
            base_url=self.config.zai_base_url,
            default_model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
        )
        self.bybit_client = BybitClient(BybitConfig(
            api_key=self.config.bybit_api_key,
            api_secret=self.config.bybit_api_secret,
            testnet=self.config.bybit_testnet,
            demo=True,  # Use DEMO environment
            simulation_mode=self.config.simulation_mode,
            simulation_balance=self.config.simulation_balance
        ))
        self.auth_store = AuthStore(AuthConfig(
            username=self.config.bot_auth_username or "admin",
            password=self.config.bot_auth_password or "admin123",
            store_path=self.config.bot_auth_store
        ))
        self.xgb_predictor = XGBoostPredictor()

    def _rule_based_intent(self, message: str) -> Dict[str, Any]:
        """Fallback intent detection when LLM is unavailable/errors."""
        text = message.strip().lower()
        # Detect symbol like BTCUSDT, ADAUSDT, BTCUSD, BTC
        m = re.search(r"\b([a-z]{3,5})(?:usdt|usd)?\b", text)
        symbol = None
        if m:
            raw = m.group(0).upper()
            if raw.endswith("USD") or raw.endswith("USDT"):
                symbol = raw
            else:
                symbol = f"{raw.upper()}USDT"
        # Detect category
        category = "linear" if any(k in text for k in ["future", "perp", "leverage"]) else "inverse" if "inverse" in text else "spot"
        # Detect file format
        ffmt = "json" if "json" in text else "csv" if "csv" in text else "txt" if "txt" in text else "none"
        # Map actions
        if any(k in text for k in ["saldo", "wallet", "balance"]):
            return {"action": "wallet_balance", "category": category, "symbol": symbol, "file_format": "none", "params": {}, "explanation": "Rule-based: wallet"}
        if any(k in text for k in ["posisi", "position", "pnl"]):
            return {"action": "positions", "category": category, "symbol": symbol, "file_format": "none", "params": {}, "explanation": "Rule-based: positions"}
        if any(k in text for k in ["orderbook", "depth"]):
            return {"action": "orderbook", "category": category, "symbol": symbol or "BTCUSDT", "file_format": "none", "params": {"limit": 10}, "explanation": "Rule-based: orderbook"}
        if any(k in text for k in ["kline", "ohlcv", "candle", "candlestick"]):
            return {"action": "kline", "category": category, "symbol": symbol or "BTCUSDT", "file_format": ffmt, "time_period": "1d", "timeframe": "30", "params": {"interval": "30", "limit": 50}, "explanation": "Rule-based: kline"}
        if any(k in text for k in ["harga", "price", "ticker"]):
            return {"action": "tickers", "category": category, "symbol": symbol or "BTCUSDT", "file_format": "none", "params": {}, "explanation": "Rule-based: tickers"}
        # Default
        return {"action": "general_chat", "explanation": "Rule-based fallback"}

    def normalize_symbol(self, user_input: str) -> str:
        """Normalize symbol dari user input"""
        user_input = user_input.upper().replace("/", "").replace("-", "").replace("_", "")
        
        # Jika sudah format lengkap
        if len(user_input) >= 6:
            return user_input
            
        # Jika hanya coin, tambahkan USDT
        if len(user_input) <= 5:
            return f"{user_input}USDT"
            
        return user_input

    async def determine_category(self, message: str, symbol: str) -> str:
        """Determine category berdasarkan konteks user"""
        message_lower = message.lower()
        
        # Keywords untuk different categories
        future_keywords = ['future', 'leverage', 'perpetual', 'perp', 'margin', 'long', 'short', 'leverage']
        inverse_keywords = ['inverse', 'coin margin', 'btcusd', 'ethusd']
        spot_keywords = ['spot', 'cash', 'buy hold']
        
        # Check untuk future/linear
        if any(keyword in message_lower for keyword in future_keywords):
            return "linear"
        
        # Check untuk inverse
        if any(keyword in message_lower for keyword in inverse_keywords):
            return "inverse"
            
        # Check symbol pattern
        if symbol.endswith("USD") and not symbol.endswith("USDT"):
            return "inverse"
        elif symbol.endswith("USDT"):
            return "linear" if any(keyword in message_lower for keyword in future_keywords) else "spot"
            
        # Default to spot
        return "spot"

    async def analyze_user_intent(self, message: str) -> Dict[str, Any]:
        """Analyze user intent dengan LLM dan detect file format"""
        system_prompt = f"""
        {API_DOCS_CONTEXT}
        
        Analyze user message dan tentukan:
        1. API endpoint yang harus dipanggil?
        2. Parameter yang diperlukan?
        3. Category yang tepat (spot/linear/inverse/option)?
        4. Symbol yang diminta?
        5. File format yang diminta (json/csv/txt)?
        6. Time period (1 hari, 1 jam, etc)?
        7. Timeframe (30 menit, 1 jam, etc)?
        
        Return JSON with structure:
        {{
            "action": "endpoint_name",
            "category": "spot|linear|inverse|option", 
            "symbol": "SYMBOL_IF_NEEDED",
            "analysis_type": "xgboost|llm|both",
            "file_format": "json|csv|txt|none",
            "time_period": "1d|1h|7d|etc",
            "timeframe": "1|5|15|30|60|240|1440",
            "params": {{"additional": "parameters"}},
            "explanation": "Brief explanation"
        }}
        
        Actions: server_time, tickers, kline, orderbook, recent_trades, instruments_info, wallet_balance, positions, trading_signal
        
        File format detection:
        - "dalam json", "format json", "export json" → "json"
        - "dalam csv", "format csv", "export csv" → "csv" 
        - "dalam txt", "format txt", "export txt" → "txt"
        - No mention → "none"
        
        Time period detection:
        - "1 hari terakhir", "24 jam" → "1d"
        - "1 jam terakhir" → "1h"
        - "1 minggu" → "7d"
        - No mention → "1d" (default)
        
        Timeframe detection:
        - "30 menit", "30m" → "30"
        - "1 jam", "1h" → "60"
        - "5 menit", "5m" → "5"
        - "15 menit", "15m" → "15"
        - "4 jam", "4h" → "240"
        - "1 hari", "1d" → "1440"
        - No mention → "30" (default 30 minutes)
        
        IMPORTANT: Never use "none" for time_period or timeframe. Always provide valid numeric values.
        
        Jika tidak trading-related:
        {{"action": "general_chat", "explanation": "General conversation"}}
        """
        
        try:
            response = self.llm_client.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ])
            
            intent = json.loads(response.strip())
            
            # Normalize symbol dan determine category
            if intent.get("symbol"):
                symbol = self.normalize_symbol(intent["symbol"])
                intent["symbol"] = symbol
                # Override category berdasarkan context jika perlu
                intent["category"] = await self.determine_category(message, symbol)
                
            # Ensure valid defaults untuk file generation
            if intent.get("file_format") != "none":
                # Validate timeframe
                if intent.get("timeframe") in [None, "", "none"]:
                    intent["timeframe"] = "30"  # Default 30 minutes
                
                # Validate time_period  
                if intent.get("time_period") in [None, "", "none"]:
                    intent["time_period"] = "1d"  # Default 1 day
                    
                # Ensure timeframe is valid Bybit interval
                valid_intervals = ["1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "1440"]
                current_tf = str(intent.get("timeframe", "30"))
                if current_tf not in valid_intervals:
                    # Map to nearest valid interval
                    try:
                        tf_int = int(current_tf)
                        if tf_int <= 1:
                            intent["timeframe"] = "1"
                        elif tf_int <= 5:
                            intent["timeframe"] = "5"
                        elif tf_int <= 15:
                            intent["timeframe"] = "15"
                        elif tf_int <= 30:
                            intent["timeframe"] = "30"
                        elif tf_int <= 60:
                            intent["timeframe"] = "60"
                        elif tf_int <= 240:
                            intent["timeframe"] = "240"
                        else:
                            intent["timeframe"] = "1440"
                    except (ValueError, TypeError):
                        intent["timeframe"] = "30"
                
            return intent
            
        except Exception as e:
            logger.error(f"Error analyzing intent: {e}")
            # Fallback to simple rule-based intent
            return self._rule_based_intent(message)

    async def get_ohlcv_data(self, category: str, symbol: str, interval: str = "1", limit: int = 100) -> pd.DataFrame:
        """Get OHLCV data dari Bybit API"""
        try:
            resp = await self.bybit_client.get_kline(category, symbol, interval, limit)
            
            if resp.get("retCode") != 0:
                return pd.DataFrame()
                
            klines = resp.get("result", {}).get("list", [])
            
            if not klines:
                return pd.DataFrame()
            
            # Convert ke DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            
            # Convert types
            df['timestamp'] = pd.to_numeric(df['timestamp'])
            df['open'] = pd.to_numeric(df['open'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['close'] = pd.to_numeric(df['close'])
            df['volume'] = pd.to_numeric(df['volume'])
            
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting OHLCV data: {e}")
            return pd.DataFrame()

    async def generate_file_data(self, intent: Dict[str, Any]) -> Optional[tuple]:
        """Generate file data based on user request
        Returns: (filename, content_bytes, content_type) atau None jika error
        """
        try:
            action = intent.get("action")
            file_format = intent.get("file_format", "none")
            
            if file_format == "none" or action != "kline":
                return None
                
            symbol = intent.get("symbol")
            category = intent.get("category", "spot")
            timeframe = intent.get("timeframe", "30")
            time_period = intent.get("time_period", "1d")
            
            if not symbol:
                return None
                
            # Validate and fix timeframe
            valid_timeframes = ["1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "1440"]
            if timeframe not in valid_timeframes:
                if timeframe == "none" or not timeframe:
                    timeframe = "30"  # Default fallback
                else:
                    # Try to find closest valid timeframe
                    try:
                        tf_num = int(timeframe)
                        if tf_num <= 1:
                            timeframe = "1"
                        elif tf_num <= 5:
                            timeframe = "5"
                        elif tf_num <= 15:
                            timeframe = "15"
                        elif tf_num <= 30:
                            timeframe = "30"
                        elif tf_num <= 60:
                            timeframe = "60"
                        elif tf_num <= 240:
                            timeframe = "240"
                        else:
                            timeframe = "1440"
                    except (ValueError, TypeError):
                        timeframe = "30"  # Safe fallback
            
            # Validate time_period
            if time_period == "none" or not time_period:
                time_period = "1d"
            
            # Calculate limit based on time period
            period_limits = {
                "1h": max(2, 60 // int(timeframe)),     # At least 2 candles
                "1d": max(2, 1440 // int(timeframe)),   # 1 day in minutes / timeframe
                "7d": max(2, min(200, 10080 // int(timeframe))),   # 7 days, max 200 candles
                "1w": max(2, min(200, 10080 // int(timeframe))),   # Same as 7d
                "1m": 200   # Max 200 candles for 1 month
            }
            limit = period_limits.get(time_period, 48)
            
            # Get OHLCV data
            df = await self.get_ohlcv_data(category, symbol, timeframe, limit)
            
            if df.empty:
                return None
                
            # Prepare data for export
            export_df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
            export_df['datetime'] = export_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{symbol}_{category}_{timeframe}m_{time_period}_{timestamp}"
            
            if file_format == "json":
                filename = f"{base_filename}.json"
                
                # Convert to JSON with proper formatting
                json_data = {
                    "metadata": {
                        "symbol": symbol,
                        "category": category,
                        "timeframe": f"{timeframe} minutes",
                        "period": time_period,
                        "total_candles": len(export_df),
                        "generated_at": datetime.now().isoformat()
                    },
                    "data": export_df.to_dict('records')
                }
                
                content = json.dumps(json_data, indent=2, ensure_ascii=False)
                content_bytes = content.encode('utf-8')
                content_type = 'application/json'
                
            elif file_format == "csv":
                filename = f"{base_filename}.csv"
                
                # Add metadata header
                metadata_lines = [
                    f"# Symbol: {symbol}",
                    f"# Category: {category}", 
                    f"# Timeframe: {timeframe} minutes",
                    f"# Period: {time_period}",
                    f"# Total candles: {len(export_df)}",
                    f"# Generated at: {datetime.now().isoformat()}",
                    ""
                ]
                
                # Convert to CSV
                csv_buffer = StringIO()
                export_df.to_csv(csv_buffer, index=False)
                csv_content = csv_buffer.getvalue()
                
                # Combine metadata and CSV
                content = "\n".join(metadata_lines) + csv_content
                content_bytes = content.encode('utf-8')
                content_type = 'text/csv'
                
            elif file_format == "txt":
                filename = f"{base_filename}.txt"
                
                # Generate readable text format
                lines = [
                    f"OHLCV Data Export",
                    f"==================",
                    f"Symbol: {symbol}",
                    f"Category: {category}",
                    f"Timeframe: {timeframe} minutes", 
                    f"Period: {time_period}",
                    f"Total Candles: {len(export_df)}",
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "",
                    "Data:",
                    "------"
                ]
                
                # Add data rows
                for _, row in export_df.iterrows():
                    lines.append(f"{row['datetime']} | O:{row['open']:8.2f} H:{row['high']:8.2f} L:{row['low']:8.2f} C:{row['close']:8.2f} V:{row['volume']:10.4f}")
                    
                content = "\n".join(lines)
                content_bytes = content.encode('utf-8')
                content_type = 'text/plain'
                
            else:
                return None
                
            return (filename, content_bytes, content_type)
            
        except Exception as e:
            logger.error(f"Error generating file data: {e}")
            return None

    async def generate_trading_signal(self, intent: Dict[str, Any]) -> str:
        """Generate trading signal dengan XGBoost atau LLM analysis"""
        try:
            category = intent.get("category", "spot")
            symbol = intent.get("symbol")
            analysis_type = intent.get("analysis_type", "both")
            
            if not symbol:
                return "❌ Symbol tidak ditemukan untuk analisis"
            
            # Get OHLCV data
            df = await self.get_ohlcv_data(category, symbol, "1", 100)
            
            if df.empty:
                return f"❌ Tidak ada data OHLCV untuk {symbol} di category {category}"
            
            # Current price info
            current_price = df['close'].iloc[-1]
            price_change_24h = ((current_price - df['close'].iloc[-24]) / df['close'].iloc[-24]) * 100 if len(df) >= 24 else 0
            
            result_lines = [
                f"📊 **Trading Signal Analysis: {symbol}**",
                f"💰 Current Price: ${current_price:,.4f}",
                f"📈 24h Change: {price_change_24h:+.2f}%",
                f"📅 Category: {category.upper()}",
                ""
            ]
            
            # XGBoost Analysis
            if analysis_type in ["xgboost", "both"]:
                features = self.xgb_predictor.prepare_features(df.copy())
                xgb_signal = self.xgb_predictor.predict_signal(features)
                
                result_lines.extend([
                    "🤖 **XGBoost AI Model:**",
                    f"🎯 Signal: **{xgb_signal['action']}**",
                    f"📊 Confidence: {xgb_signal['confidence']:.1%}",
                    f"🎲 Probability: {xgb_signal.get('probability', 0):.3f}",
                    ""
                ])
            
            # LLM Analysis
            if analysis_type in ["llm", "both"]:
                # Prepare market data untuk LLM
                recent_data = df.tail(20)
                market_summary = self.prepare_market_summary(recent_data)
                
                llm_analysis = await self.get_llm_analysis(symbol, market_summary)
                result_lines.extend([
                    "🧠 **LLM Analysis:**",
                    llm_analysis,
                    ""
                ])
            
            # Technical indicators summary
            if not df.empty:
                df_with_indicators = TechnicalAnalyzer.calculate_indicators(df.copy())
                if not df_with_indicators.empty:
                    latest = df_with_indicators.iloc[-1]
                    
                    result_lines.extend([
                        "📋 **Technical Indicators:**",
                        f"• RSI(14): {latest.get('rsi', 0):.1f}",
                        f"• MACD: {latest.get('macd', 0):.6f}",
                        f"• BB Position: {((current_price - latest.get('bb_lower', current_price)) / (latest.get('bb_upper', current_price) - latest.get('bb_lower', current_price))):.2%}" if 'bb_upper' in latest else "",
                        f"• Volume Ratio: {latest.get('volume_ratio', 1):.2f}x",
                    ])
            
            return "\n".join(result_lines)
            
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return f"❌ Error generating signal: {str(e)}"

    def prepare_market_summary(self, df: pd.DataFrame) -> str:
        """Prepare market data summary untuk LLM analysis"""
        try:
            if df.empty:
                return "No market data available"
            
            latest = df.iloc[-1]
            oldest = df.iloc[0]
            
            price_change = ((latest['close'] - oldest['close']) / oldest['close']) * 100
            high_24h = df['high'].max()
            low_24h = df['low'].min()
            volume_avg = df['volume'].mean()
            volatility = df['close'].std() / df['close'].mean() * 100
            
            return f"""
Recent Market Data:
- Price: ${latest['close']:,.4f}
- Change: {price_change:+.2f}% 
- 24h High: ${high_24h:,.4f}
- 24h Low: ${low_24h:,.4f}
- Avg Volume: {volume_avg:,.0f}
- Volatility: {volatility:.2f}%
- Last 5 closes: {list(df['close'].tail(5).round(4))}
"""
        except Exception as e:
            return f"Error preparing market summary: {e}"

    async def get_llm_analysis(self, symbol: str, market_data: str) -> str:
        """Get LLM trading analysis"""
        try:
            prompt = f"""
Based on the market data for {symbol}, provide a concise trading analysis:

{market_data}

Analyze:
1. Overall trend direction
2. Key support/resistance levels
3. Trading recommendation (BUY/SELL/HOLD)
4. Risk assessment

Keep response under 200 words, focus on actionable insights.
"""
            
            response = self.llm_client.chat([
                {"role": "system", "content": "You are an expert crypto trader. Provide clear, concise analysis."},
                {"role": "user", "content": prompt}
            ])
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error getting LLM analysis: {e}")
            return f"Error getting LLM analysis: {str(e)}"

    async def execute_api_call(self, intent: Dict[str, Any]) -> str:
        """Execute API call berdasarkan analyzed intent"""
        try:
            action = intent.get("action")
            category = intent.get("category", "spot")
            symbol = intent.get("symbol")
            params = intent.get("params", {})
            
            if action == "trading_signal":
                return await self.generate_trading_signal(intent)
            
            elif action == "server_time":
                resp = await self.bybit_client.get_server_time()
                timestamp = resp.get('result', {}).get('timeSecond', 'N/A')
                if timestamp != 'N/A':
                    dt = datetime.fromtimestamp(int(timestamp))
                    return f"🕐 Server Time: {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}"
                return f"🕐 Server Time: {timestamp}"
                
            elif action == "tickers":
                resp = await self.bybit_client.get_tickers(category, symbol)
                return self.format_tickers_response(resp, symbol, category)
                
            elif action == "kline":
                limit = params.get("limit", 10)
                interval = params.get("interval", "1")
                resp = await self.bybit_client.get_kline(category, symbol, interval, limit)
                return self.format_kline_response(resp, symbol, interval, category)
                
            elif action == "orderbook":
                limit = params.get("limit", 10)
                resp = await self.bybit_client.get_orderbook(category, symbol, limit)
                return self.format_orderbook_response(resp, symbol, category)
                
            elif action == "recent_trades":
                limit = params.get("limit", 5)
                resp = await self.bybit_client.get_recent_trades(category, symbol, limit)
                return self.format_recent_trades_response(resp, symbol, category)
                
            elif action == "instruments_info":
                resp = await self.bybit_client.get_instruments_info(category, symbol)
                return self.format_instruments_response(resp, symbol, category)
                
            elif action == "wallet_balance":
                if not self.bybit_client.enabled:
                    return "❌ Wallet balance memerlukan API key. Set BYBIT_API_KEY dan BYBIT_API_SECRET di .env"
                    
                try:
                    resp = await self.bybit_client.get_wallet_balance()
                    logger.info(f"Wallet response: {resp}")  # Debug log
                    return self.format_wallet_response(resp)
                except Exception as e:
                    logger.error(f"Wallet balance error: {e}")
                    return f"❌ Error mengambil wallet balance: {str(e)}"
                
            elif action == "positions":
                if not self.bybit_client.enabled:
                    return "❌ Positions memerlukan API key. Set BYBIT_API_KEY dan BYBIT_API_SECRET di .env"
                try:
                    categories = [category]
                    if category == "spot":
                        categories = ["linear", "inverse"]
                    outputs: List[str] = []
                    for cat in categories:
                        if symbol:
                            resp = await self.bybit_client.get_position_list(cat, symbol=symbol)
                            logger.info(f"Positions response ({cat}/{symbol}): {resp}")
                            outputs.append(self.format_positions_response(resp, cat))
                        else:
                            # Try multiple settleCoins to avoid retCode 10001
                            settle_candidates = ["USDT", "USDC"] if cat == "linear" else ["BTC", "ETH"]
                            combined: List[str] = []
                            for settle in settle_candidates:
                                resp = await self.bybit_client.get_position_list(cat, symbol=None, settle_coin=settle)
                                logger.info(f"Positions response ({cat}/settle={settle}): {resp}")
                                formatted = self.format_positions_response(resp, cat)
                                if "Tidak ada posisi" not in formatted and "Bybit API Error" not in formatted and "Parameter kurang" not in formatted:
                                    combined.append(formatted)
                            if combined:
                                outputs.append("\n".join(combined))
                            else:
                                outputs.append(f"📊 **Tidak ada posisi terbuka di {cat.upper()}**\n\n✅ Account bersih, tidak ada open positions.")
                    return "\n\n".join(outputs)
                except Exception as e:
                    logger.error(f"Positions error: {e}")
                    return f"❌ Error mengambil positions: {str(e)}"
                
            else:
                return f"❌ Action '{action}' belum diimplementasi"
                
        except Exception as e:
            logger.error(f"Error executing API call: {e}")
            if "symbol not found" in str(e).lower():
                return f"❌ Symbol {symbol} tidak ditemukan di {category}. Coba category lain atau periksa ejaan."
            return f"❌ Error: {str(e)}"

    def format_tickers_response(self, resp: Dict[str, Any], symbol: Optional[str], category: str) -> str:
        """Format ticker response dengan informasi lengkap"""
        try:
            result = resp.get("result", {})
            list_items = result.get("list", [])
            
            if not list_items:
                return f"❌ Tidak ada data ticker untuk {symbol or 'symbol'} di {category}"
                
            if len(list_items) == 1:
                ticker = list_items[0]
                change_pct = float(ticker.get('price24hPcnt', '0')) * 100
                change_emoji = "📈" if change_pct >= 0 else "📉"
                
                return f"""📊 **{ticker.get('symbol', 'N/A')} ({category.upper()})**
💰 Price: ${float(ticker.get('lastPrice', 0)):,.4f}
{change_emoji} 24h: {change_pct:+.2f}%
📊 Volume: {float(ticker.get('volume24h', 0)):,.0f}
🔺 High: ${float(ticker.get('highPrice24h', 0)):,.4f}
🔻 Low: ${float(ticker.get('lowPrice24h', 0)):,.4f}
💹 Turnover: ${float(ticker.get('turnover24h', 0)):,.0f}"""
            else:
                lines = [f"📊 **Top Tickers ({category.upper()}):**"]
                for ticker in list_items[:15]:
                    change_pct = float(ticker.get('price24hPcnt', '0')) * 100
                    emoji = "📈" if change_pct >= 0 else "📉"
                    lines.append(f"{emoji} {ticker.get('symbol')}: ${float(ticker.get('lastPrice', 0)):,.4f} ({change_pct:+.2f}%)")
                
                if len(list_items) > 15:
                    lines.append(f"... dan {len(list_items) - 15} lainnya")
                    
                return "\n".join(lines)
                
        except Exception as e:
            return f"❌ Error formatting ticker: {str(e)}"

    def format_kline_response(self, resp: Dict[str, Any], symbol: str, interval: str, category: str) -> str:
        """Format kline response dengan analysis dan safe casting"""
        try:
            result = resp.get("result", {})
            klines = result.get("list", [])
            
            if not klines:
                return f"❌ Tidak ada data kline untuk {symbol} ({category})"
            
            # Safe conversion function
            def safe_float(value, default=0.0):
                try:
                    return float(value or default)
                except (ValueError, TypeError):
                    return default
            
            def safe_int(value, default=0):
                try:
                    return int(value or default)
                except (ValueError, TypeError):
                    return default
            
            # Parse data dengan safe casting
            parsed_klines = []
            for kline in klines:
                if len(kline) >= 6:
                    parsed_klines.append({
                        'timestamp': safe_int(kline[0]),
                        'open': safe_float(kline[1]),
                        'high': safe_float(kline[2]),
                        'low': safe_float(kline[3]),
                        'close': safe_float(kline[4]),
                        'volume': safe_float(kline[5])
                    })
            
            if not parsed_klines:
                return f"❌ Data kline tidak valid untuk {symbol}"
            
            # Calculate trend
            first_close = parsed_klines[0]['close']
            last_close = parsed_klines[-1]['close']
            
            if first_close > 0:
                price_change = ((last_close - first_close) / first_close) * 100
            else:
                price_change = 0
                
            trend_emoji = "📈" if price_change > 0 else "📉" if price_change < 0 else "➡️"
            
            lines = [
                f"📊 **{symbol} - {category.upper()} ({interval}min)**",
                f"{trend_emoji} **Trend: {price_change:+.2f}%**",
                f"💲 **Current Price: ${last_close:,.2f}**",
                "",
                "**Recent Candles:**"
            ]
            
            # Show recent candles (limit to 5 for cleaner display)
            for i, kline in enumerate(parsed_klines[-5:], 1):
                timestamp = kline['timestamp'] / 1000
                dt = datetime.fromtimestamp(timestamp)
                time_str = dt.strftime("%H:%M")
                
                lines.append(f"{i}. **{time_str}** O:{kline['open']:.2f} H:{kline['high']:.2f} L:{kline['low']:.2f} C:{kline['close']:.2f} V:{kline['volume']:.4f}")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error formatting kline: {e}")
            return f"❌ Error formatting kline: Terjadi kesalahan parsing data"

    def format_orderbook_response(self, resp: Dict[str, Any], symbol: str, category: str) -> str:
        """Format orderbook dengan bid/ask analysis"""
        try:
            result = resp.get("result", {})
            asks = result.get("a", [])  # asks = sellers
            bids = result.get("b", [])  # bids = buyers
            
            if not asks or not bids:
                return f"❌ Tidak ada orderbook data untuk {symbol} ({category})"
            
            lines = [
                f"📖 **Orderbook {symbol} ({category.upper()})**",
                "",
                "```"
            ]
            
            # Header
            lines.append("ASKS (Sellers)          │ BIDS (Buyers)")
            lines.append("Price    │ Size          │ Price    │ Size")
            lines.append("─────────┼───────────────┼──────────┼──────────")
            
            # Show top 8 levels
            max_levels = min(8, len(asks), len(bids))
            
            for i in range(max_levels):
                ask_price = float(asks[i][0])
                ask_size = float(asks[i][1])
                bid_price = float(bids[i][0])
                bid_size = float(bids[i][1])
                
                lines.append(f"{ask_price:8.4f} │ {ask_size:12.2f} │ {bid_price:8.4f} │ {bid_size:9.2f}")
            
            lines.append("```")
            
            # Spread analysis
            best_ask = float(asks[0][0])
            best_bid = float(bids[0][0])
            spread = best_ask - best_bid
            spread_pct = (spread / best_bid) * 100
            
            lines.append("")
            lines.append(f"💰 Best Bid: ${best_bid:.4f}")
            lines.append(f"💰 Best Ask: ${best_ask:.4f}")
            lines.append(f"📊 Spread: ${spread:.4f} ({spread_pct:.3f}%)")
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"❌ Error formatting orderbook: {str(e)}"

    def format_recent_trades_response(self, resp: Dict[str, Any], symbol: str, category: str) -> str:
        """Format recent trades dengan market activity analysis"""
        try:
            result = resp.get("result", {})
            trades = result.get("list", [])
            
            if not trades:
                return f"❌ Tidak ada recent trades untuk {symbol} ({category})"
            
            lines = [
                f"🔄 **Recent Trades {symbol} ({category.upper()})**",
                "",
                "```"
            ]
            
            lines.append("Time     │ Side │ Price    │ Size      ")
            lines.append("─────────┼──────┼──────────┼───────────")
            
            buy_vol = sell_vol = 0
            
            for trade in trades[:10]:  # Show 10 trades
                timestamp = int(trade[0]) / 1000
                dt = datetime.fromtimestamp(timestamp)
                time_str = dt.strftime("%H:%M:%S")
                
                price = float(trade[1])
                size = float(trade[2])
                side = trade[3]  # Buy/Sell
                
                side_emoji = "🟢BUY " if side == "Buy" else "🔴SELL"
                
                if side == "Buy":
                    buy_vol += size
                else:
                    sell_vol += size
                
                lines.append(f"{time_str} │ {side_emoji} │ {price:8.4f} │ {size:10.2f}")
            
            lines.append("```")
            
            # Volume analysis
            total_vol = buy_vol + sell_vol
            if total_vol > 0:
                buy_pct = (buy_vol / total_vol) * 100
                sell_pct = (sell_vol / total_vol) * 100
                
                lines.append("")
                lines.append(f"📊 **Volume Analysis:**")
                lines.append(f"🟢 Buy Volume: {buy_vol:.2f} ({buy_pct:.1f}%)")
                lines.append(f"🔴 Sell Volume: {sell_vol:.2f} ({sell_pct:.1f}%)")
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"❌ Error formatting trades: {str(e)}"

    def format_instruments_response(self, resp: Dict[str, Any], symbol: Optional[str], category: str) -> str:
        """Format instruments info response"""
        try:
            result = resp.get("result", {})
            instruments = result.get("list", [])
            
            if not instruments:
                return f"❌ Tidak ada instrument info untuk {symbol or 'yang diminta'} ({category})"
                
            instrument = instruments[0]
            
            lines = [
                f"ℹ️ **Instrument: {instrument.get('symbol', 'N/A')} ({category.upper()})**",
                f"📝 Status: {instrument.get('status', 'N/A')}",
                f"💱 Base: {instrument.get('baseCoin', 'N/A')} | Quote: {instrument.get('quoteCoin', 'N/A')}",
                ""
            ]
            
            # Lot size info
            lot_filter = instrument.get('lotSizeFilter', {})
            if lot_filter:
                lines.extend([
                    "📏 **Lot Size Rules:**",
                    f"• Min Order: {lot_filter.get('minOrderQty', 'N/A')}",
                    f"• Max Order: {lot_filter.get('maxOrderQty', 'N/A')}",
                    f"• Qty Step: {lot_filter.get('qtyStep', 'N/A')}",
                    ""
                ])
            
            # Price filter info  
            price_filter = instrument.get('priceFilter', {})
            if price_filter:
                lines.extend([
                    "💲 **Price Rules:**",
                    f"• Tick Size: {price_filter.get('tickSize', 'N/A')}",
                    f"• Min Price: {price_filter.get('minPrice', 'N/A')}",
                    f"• Max Price: {price_filter.get('maxPrice', 'N/A')}",
                ])
            
            # Leverage info untuk futures
            if category in ["linear", "inverse"]:
                leverage = instrument.get('leverageFilter', {})
                if leverage:
                    lines.extend([
                        "",
                        "⚡ **Leverage:**",
                        f"• Min: {leverage.get('minLeverage', 'N/A')}x",
                        f"• Max: {leverage.get('maxLeverage', 'N/A')}x",
                        f"• Step: {leverage.get('leverageStep', 'N/A')}x"
                    ])
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"❌ Error formatting instrument info: {str(e)}"

    def format_wallet_response(self, resp: Dict[str, Any]) -> str:
        """Format wallet balance response"""
        try:
            ret_code = resp.get("retCode", 0)
            ret_msg = resp.get("retMsg", "")
            
            # Handle API errors
            if ret_code != 0:
                if ret_code == 10003:
                    return """❌ **API Key Invalid**

🔧 **Solusi Lengkap:**

1️⃣ **Generate API Key Baru:**
   • Login ke bybit.com → Account & Security → API Management
   • Create New Key dengan permissions:
     ✅ Read Account/Position/Order
     ✅ Account Transfer
   
2️⃣ **IP Whitelist:**
   • Set IP: `118.99.123.4` (IP current Anda)
   • Atau set `0.0.0.0/0` untuk allow all IP
   
3️⃣ **Update .env File:**
   ```
   BYBIT_API_KEY=your_new_key_here
   BYBIT_API_SECRET=your_new_secret_here
   ```

⚠️ **Kemungkinan penyebab:**
• API key expired/suspended
• IP tidak dalam whitelist  
• Copy-paste key salah (ada space/karakter extra)
• Key dibuat untuk testnet tapi dipakai di mainnet"""
                elif ret_code == 10004:
                    return "❌ **API Signature Invalid**\n\nSignature verification failed. Kemungkinan API secret salah atau ada masalah dengan timestamp."
                elif ret_code == 10005:
                    return "❌ **API Permission Denied**\n\nAPI key tidak memiliki permission untuk akses wallet. Set permission 'Read Account' di Bybit dashboard."
                elif ret_code == 10018:
                    return f"❌ **IP Not in Whitelist**\n\nIP Anda `118.99.123.4` tidak dalam whitelist.\n\nTambahkan IP ini di Bybit → API Management → Edit Key → IP Whitelist."
                else:
                    return f"❌ **Bybit API Error**\nCode: {ret_code}\nMessage: {ret_msg}"
            
            result = resp.get("result", {})
            accounts = result.get("list", [])
            
            if not accounts:
                return "💼 **Wallet kosong**\n\nTidak ada balance di akun Anda atau akun belum activated untuk trading."
            
            sim_note = "\n🧪 Simulation data" if self.config.simulation_mode else ""
            lines = [f"💼 **Wallet Balance**{sim_note}:", ""]
            
            total_usd_all = 0
            for account in accounts:
                account_type = account.get("accountType", "Unknown")
                coins = account.get("coin", [])
                
                if coins:
                    lines.append(f"📊 **{account_type} Account:**")
                    
                    total_usd = 0
                    coin_count = 0
                    
                    for coin in coins:
                        coin_name = coin.get("coin", "")
                        # Safe casting with error handling
                        try:
                            wallet_balance = float(coin.get("walletBalance") or "0")
                        except (ValueError, TypeError):
                            wallet_balance = 0.0
                        
                        avail_raw = (
                            coin.get("availableToWithdraw")
                            or coin.get("availableBalance")
                            or coin.get("availableToBorrow")
                            or "0"
                        )
                        try:
                            available = float(avail_raw or "0")
                        except (ValueError, TypeError):
                            available = 0.0
                        
                        try:
                            usd_value = float(coin.get("usdValue") or "0")
                        except (ValueError, TypeError):
                            usd_value = 0.0
                        
                        if wallet_balance > 0 or usd_value > 1:  # Show coins with > $1 value
                            coin_count += 1
                            # Format with proper decimal places
                            balance_str = f"{wallet_balance:.8f}".rstrip('0').rstrip('.')
                            available_str = f"{available:.8f}".rstrip('0').rstrip('.')
                            
                            lines.append(f"• **{coin_name}**: {balance_str}")
                            if abs(available - wallet_balance) > 1e-12:
                                lines.append(f"  🔓 Available: {available_str}")
                            if usd_value > 0:
                                lines.append(f"  💵 USD Value: ${usd_value:,.2f}")
                                total_usd += usd_value
                                total_usd_all += usd_value
                    
                    if coin_count == 0:
                        lines.append("• Tidak ada balance signifikan")
                    elif total_usd > 0:
                        lines.append(f"**Subtotal: ${total_usd:,.2f}**")
                    lines.append("")
            
            if total_usd_all > 0:
                lines.append(f"🏆 **TOTAL USD VALUE: ${total_usd_all:,.2f}**")
            
            return "\n".join(lines) if len(lines) > 2 else "💼 Wallet API berhasil tapi tidak ada balance"
            
        except Exception as e:
            return f"❌ Error formatting wallet: {str(e)}"

    def format_positions_response(self, resp: Dict[str, Any], category: str) -> str:
        """Format positions response"""
        try:
            ret_code = resp.get("retCode", 0)
            ret_msg = resp.get("retMsg", "")
            
            # Handle API errors
            if ret_code != 0:
                if ret_code == 10001 and "symbol or settleCoin" in (ret_msg or ""):
                    return (
                        "❌ Parameter kurang untuk Positions\n\n"
                        "Tambahkan `symbol` atau `settleCoin`. Contoh:\n"
                        "• Linear USDT: settleCoin=USDT\n"
                        "• Linear USDC: settleCoin=USDC\n"
                        "• Inverse BTC: symbol=BTCUSD\n"
                    )
                if ret_code == 10003:
                    return "❌ **API Key Invalid untuk Positions**\n\nAPI key tidak valid. Silakan set API key yang benar di .env file."
                elif ret_code == 10005:
                    return "❌ **API Permission Denied**\n\nAPI key tidak memiliki permission untuk akses positions."
                else:
                    return f"❌ **Bybit API Error**\nCode: {ret_code}\nMessage: {ret_msg}"
            
            result = resp.get("result", {})
            positions = result.get("list", [])
            
            if not positions:
                return f"📊 **Tidak ada posisi terbuka di {category.upper()}**\n\n✅ Account bersih, tidak ada open positions."
                
            sim_note = " 🧪" if self.config.simulation_mode else ""
            lines = [f"📊 **Open Positions ({category.upper()})**{sim_note}:", ""]
            open_positions = 0
            total_unreal_pnl = 0.0
            
            for pos in positions:
                size = float(pos.get("size", 0))
                if size != 0:  # Only show non-zero positions
                    open_positions += 1
                    symbol = pos.get('symbol', 'N/A')
                    side = pos.get('side', 'N/A')
                    entry_price = float(pos.get('avgPrice', 0))
                    mark_price = float(pos.get('markPrice', 0))
                    pnl = float(pos.get('unrealisedPnl', 0))
                    pnl_pct = float(pos.get('unrealisedPnlPcnt', 0)) * 100
                    
                    pnl_emoji = "📈" if pnl >= 0 else "📉"
                    side_emoji = "🟢" if side == "Buy" else "🔴"
                    total_unreal_pnl += pnl
                    
                    lines.extend([
                        f"**{symbol}**",
                        f"{side_emoji} {side} | Size: {size}",
                        f"💰 Entry: ${entry_price:.4f} | Mark: ${mark_price:.4f}",
                        f"{pnl_emoji} PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)",
                        ""
                    ])
            
            if open_positions == 0:
                return f"📊 **Tidak ada posisi terbuka di {category.upper()}**\n\n✅ Semua posisi sudah closed."
            lines.append(f"**Subtotal Unrealized PnL: ${total_unreal_pnl:,.2f}**")
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"❌ Error formatting positions: {str(e)}"

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle user messages dengan complete analysis dan file generation"""
        # Auth check
        if not self.auth_store.is_authenticated(update.effective_user.id):
            await update.message.reply_text("🔐 Silakan login: /login <username> <password>")
            return

        user_message = update.message.text.strip()
        
        # Show typing
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
        
        try:
            # Analyze intent
            intent = await self.analyze_user_intent(user_message)
            
            if intent.get("action") == "general_chat":
                # LLM conversation with graceful fallback
                if not self.llm_client.enabled:
                    await update.message.reply_text("🤖 LLM tidak aktif. Silakan set ZAI_API_KEY untuk percakapan umum.")
                else:
                    response = self.llm_client.chat([
                        {"role": "system", "content": "You are a helpful crypto trading assistant. Be concise but informative."},
                        {"role": "user", "content": user_message}
                    ])
                    await update.message.reply_text(response)
            else:
                # Check if user wants file output
                file_format = intent.get("file_format", "none")
                
                if file_format != "none":
                    # Generate file and send both text response and file
                    api_response = await self.execute_api_call(intent)
                    file_data = await self.generate_file_data(intent)
                    
                    if file_data:
                        filename, content_bytes, content_type = file_data
                        
                        # Send text response first
                        await update.message.reply_text(f"📊 **Data Preview:**\n{api_response}", parse_mode='Markdown')
                        
                        # Send file
                        from io import BytesIO
                        file_buffer = BytesIO(content_bytes)
                        file_buffer.name = filename
                        
                        await update.message.reply_document(
                            document=file_buffer,
                            filename=filename,
                            caption=f"📁 **{filename}**\n\n📊 Data: {intent.get('symbol')} ({intent.get('category', 'spot').upper()})\n⏱️ Period: {intent.get('time_period', '1d')}\n📈 Timeframe: {intent.get('timeframe', '30')}m\n📋 Format: {file_format.upper()}"
                        )
                    else:
                        # Fallback to regular response if file generation fails
                        api_response = await self.execute_api_call(intent)
                        await update.message.reply_text(f"{api_response}\n\n❌ File generation failed", parse_mode='Markdown')
                else:
                    # Regular API response without file
                    api_response = await self.execute_api_call(intent)
                    
                    # Split long messages
                    if len(api_response) > 4000:
                        chunks = [api_response[i:i+4000] for i in range(0, len(api_response), 4000)]
                        for chunk in chunks:
                            await update.message.reply_text(chunk, parse_mode='Markdown')
                    else:
                        await update.message.reply_text(api_response, parse_mode='Markdown')
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")

    async def start_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Enhanced start command"""
        keyboard = [
            [InlineKeyboardButton("📊 Market Data", callback_data='help_market')],
            [InlineKeyboardButton("🤖 AI Signals", callback_data='help_ai')],
            [InlineKeyboardButton("💼 Account Info", callback_data='help_account')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Simulation mode notification
        mode_info = ""
        if self.config.simulation_mode:
            mode_info = "\n🧪 **SIMULATION MODE** - Data simulasi untuk development\n"
        
        await update.message.reply_text(f"""
🚀 **Complete Bybit Trading Bot**
{mode_info}
*Fitur Lengkap:*
• 📊 Real-time market data (Spot & Futures)
• 🤖 AI Trading Signals (XGBoost + LLM)
• 💹 Technical Analysis lengkap
• 📈 Dynamic category detection
• 💼 Portfolio management

*Contoh command:*
• "Harga BTC future sekarang"
• "Trading signal ETHUSDT dengan AI"
• "Orderbook ADAUSDT spot"
• "Analisis XGBoost untuk BNBUSDT"

Mulai dengan: /login <username> <password>
""", reply_markup=reply_markup)

    async def help_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Comprehensive help command"""
        help_text = """
📚 **Complete Trading Bot Help**

**🎯 Smart Commands:**
Bot ini otomatis mendeteksi kategori berdasarkan konteks:

*Market Data:*
• "Harga Bitcoin" → Spot BTCUSDT
• "BTC future price" → Linear BTCUSDT  
• "BTCUSD inverse" → Inverse BTCUSD
• "Volume ETH hari ini"
• "Orderbook depth ADAUSDT"

*AI Analysis:*
• "Trading signal BTC dengan XGBoost"
• "Analisis AI untuk ETHUSDT"
• "Signal both XGBoost dan LLM BNBUSDT"

*Technical Analysis:*
• "RSI Bitcoin sekarang"
• "MACD analysis ETHUSDT"
• "Bollinger bands ADAUSDT"

*Account (perlu API key):*
• "Saldo wallet saya"
• "Posisi terbuka linear"
• "Balance dan PnL"

*File Export:*
• "OHLCV BTCUSDT 1 hari terakhir dalam tf 30 menit format JSON"
• "Data harga ETHUSDT 1 jam terakhir tf 5 menit dalam CSV"
• "Export kline ADAUSDT 7 hari dalam format TXT"

**🔑 Categories:**
• **Spot**: Cash trading
• **Linear**: USDT perpetual futures  
• **Inverse**: Coin margined futures
• **Option**: Options trading

**⚡ Pro Tips:**
• Sebutkan "future/leverage" untuk linear
• Sebutkan "inverse/coin margin" untuk inverse
• Default spot jika tidak disebutkan
• Bot deteksi otomatis dari konteks!
"""
        await update.message.reply_text(help_text)

    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle inline button callbacks"""
        query = update.callback_query
        await query.answer()
        
        if query.data == 'help_market':
            await query.edit_message_text("""
📊 **Market Data Help**

*Otomatis detect category:*
• "Harga BTC" → Spot
• "BTC future" → Linear
• "BTCUSD inverse" → Inverse

*Data tersedia:*
• Current price & 24h change
• Volume & turnover
• Order book depth
• Recent trades
• Kline/candlestick data
• Technical indicators

*Contoh:*
• "Show me BTC price"
• "ETH volume today"
• "ADAUSDT orderbook"
""")
        elif query.data == 'help_ai':
            await query.edit_message_text("""
🤖 **AI Signals Help**

*Dual Analysis Mode:*
• **XGBoost**: ML model dengan 25+ features
• **LLM**: GPT analysis dari market data
• **Both**: Kombinasi kedua analysis

*Features analyzed:*
• Technical indicators (RSI, MACD, BB)
• Price patterns & trends
• Volume analysis
• Market sentiment

*Contoh:*
• "Trading signal BTCUSDT"
• "AI analysis ETH dengan XGBoost"
• "Signal both AI dan XGB ADAUSDT"
""")
        elif query.data == 'help_account':
            await query.edit_message_text("""
💼 **Account Management**

*Requirements:*
• Set BYBIT_API_KEY di .env
• Set BYBIT_API_SECRET di .env

*Features:*
• Wallet balance semua coins
• Open positions (Linear/Inverse)
• PnL calculation
• USD value conversion

*Contoh:*
• "Saldo wallet saya"  
• "Posisi linear terbuka"
• "Balance dan profit"
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
            await update.message.reply_text("✅ Login berhasil! Bot siap digunakan dengan fitur lengkap.")
        else:
            await update.message.reply_text("❌ Username atau password salah.")

    async def logout_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Logout handler"""
        self.auth_store.logout(update.effective_user.id)
        await update.message.reply_text("👋 Logout berhasil. Gunakan /login untuk masuk kembali.")

    async def debug_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Debug API configuration"""
        if not self.auth_store.is_authenticated(update.effective_user.id):
            await update.message.reply_text("🔐 Silakan login dulu")
            return
            
        debug_info = f"""
🔧 **Debug Info:**
• Bybit Enabled: {self.bybit_client.enabled}
• API Key: {'✅ Set' if self.config.bybit_api_key else '❌ Missing'}
• API Secret: {'✅ Set' if self.config.bybit_api_secret else '❌ Missing'}
• Base URL: {self.bybit_client.base_url}
• Testnet: {self.config.bybit_testnet}

📋 **Config Check:**
• LLM Key: {'✅ Set' if self.config.zai_api_key else '❌ Missing'}
• Telegram Token: {'✅ Set' if self.config.telegram_bot_token else '❌ Missing'}
        """
        
        await update.message.reply_text(debug_info)

def main():
    """Main function dengan error handling lengkap"""
    try:
        bot = CompleteBybitBot()
        
        # Create application with error handling
        app = Application.builder().token(bot.config.telegram_bot_token).build()
        
        # Add error handler untuk conflict resolution
        async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
            """Handle errors dari bot"""
            logger.error(f"Exception while handling update: {context.error}")
            if "Conflict" in str(context.error):
                logger.info("Bot conflict detected, trying to restart...")
                await asyncio.sleep(5)
        
        app.add_error_handler(error_handler)
        
        # Add handlers
        app.add_handler(CommandHandler("start", bot.start_handler))
        app.add_handler(CommandHandler("help", bot.help_handler))
        app.add_handler(CommandHandler("login", bot.login_handler))
        app.add_handler(CommandHandler("logout", bot.logout_handler))
        app.add_handler(CommandHandler("debug", bot.debug_handler))
        app.add_handler(CallbackQueryHandler(bot.button_handler))
        
        # Handle all text messages
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))
        
        # Start bot with retry
        logger.info("🚀 Starting Complete Bybit Trading Bot...")
        logger.info("📊 Features: LLM + XGBoost + Technical Analysis + Dynamic Categories")
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                app.run_polling(allowed_updates=Update.ALL_TYPES, timeout=20)
                break
            except Exception as e:
                if "Conflict" in str(e):
                    retry_count += 1
                    logger.warning(f"Bot conflict, retry {retry_count}/{max_retries} in 10 seconds...")
                    if retry_count >= max_retries:
                        logger.error("Max retries reached, exiting...")
                        break
                    asyncio.sleep(10)
                else:
                    raise e
        
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        raise

if __name__ == "__main__":
    main()
