#!/usr/bin/env python3
"""
Bybit MCP Server menggunakan FastAPI-MCP Framework
Server ini menyediakan akses lengkap ke Bybit V5 API dengan konteks dokumentasi terintegrasi
"""

import os
import sys
import json
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path

# Add src directory to path
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from fastapi_mcp import FastApiMCP

try:
    # Import from src directory
    from bybit_client import BybitClient, BybitConfig
    from config import get_config
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure src directory contains required modules")
    sys.exit(1)

# Load API documentation untuk konteks
def load_api_docs() -> str:
    """Load API documentation sebagai context"""
    try:
        api_docs_path = Path(__file__).parent.parent / "api-docs.txt"
        with open(api_docs_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Could not load API docs: {e}")
        return "API documentation not available"

# Global variables
config = get_config()
bybit_client = BybitClient(BybitConfig(
    api_key=config.bybit_api_key,
    api_secret=config.bybit_api_secret,
    testnet=config.bybit_testnet,
    public_only=config.public_only
))
api_docs_context = load_api_docs()

# Create FastAPI app
app = FastAPI(
    title="Bybit V5 API MCP Server",
    description="MCP Server for Bybit V5 API dengan konteks dokumentasi lengkap",
    version="2.0.0"
)

# Response Models
class BybitResponse(BaseModel):
    success: bool
    data: Dict[str, Any] = {}
    error: Optional[str] = None
    api_context: Optional[str] = None

class APIDocsResponse(BaseModel):
    success: bool
    data: Dict[str, Any] = {}
    section: Optional[str] = None
    available_sections: List[str] = []

# === ROOT ENDPOINT ===

@app.get("/", response_model=Dict[str, Any], operation_id="get_server_info", tags=["Server Info"])
async def get_server_info():
    """
    Get server information and available endpoints
    
    This endpoint provides a summary of all available endpoints and their categories
    """
    return {
        "server": "Multi-Exchange Trading MCP Server",
        "version": "2.0.0",
        "description": "MCP Server for Bybit V5 API + Multi-Exchange integration",
        "endpoints": {
            "bybit_public": {
                "description": "Bybit public market data endpoints (no authentication required)",
                "endpoints": [
                    "GET /bybit/tickers - Get ticker information",
                    "GET /bybit/kline - Get kline/candlestick data", 
                    "GET /bybit/orderbook - Get order book data",
                    "GET /bybit/recent-trades - Get recent trade data",
                    "GET /bybit/instruments-info - Get instrument information",
                    "GET /bybit/server-time - Get Bybit server time"
                ]
            },
            "bybit_private": {
                "description": "Bybit private endpoints (requires API credentials)",
                "endpoints": [
                    "GET /bybit/wallet-balance - Get wallet balance",
                    "GET /bybit/positions - Get position list",
                    "POST /bybit/order - Create new order"
                ]
            },
            "multi_exchange": {
                "description": "Multi-exchange public data endpoints",
                "endpoints": [
                    "GET /exchanges/binance/ticker - Get Binance ticker",
                    "GET /exchanges/kucoin/ticker - Get KuCoin ticker",
                    "GET /exchanges/okx/ticker - Get OKX ticker",
                    "GET /exchanges/huobi/ticker - Get Huobi ticker",
                    "GET /exchanges/mexc/ticker - Get MEXC ticker",
                    "GET /exchanges/compare-prices - Compare prices across exchanges"
                ]
            },
            "documentation": {
                "description": "API documentation and context",
                "endpoints": [
                    "GET /api-docs - Get API documentation context"
                ]
            }
        },
        "mcp_endpoint": "/mcp",
        "docs_endpoint": "/docs",
        "config": {
            "public_only_mode": config.public_only,
            "api_docs_loaded": len(api_docs_context) > 0
        }
    }

# Helper function untuk extract endpoint context
def get_endpoint_context(endpoint_type: str) -> str:
    """Extract specific endpoint context from API docs"""
    lines = api_docs_context.split('\n')
    context_start = False
    context_lines = []
    
    # Find relevant section based on endpoint type
    section_map = {
        'market': '### 1. Market Data Endpoints',
        'order': '### 2. Order Management Endpoints',
        'position': '### 3. Position Management Endpoints',
        'account': '### 4. Account Management Endpoints',
        'asset': '### 5. Asset Management Endpoints'
    }
    
    start_marker = section_map.get(endpoint_type, '')
    if not start_marker:
        return "Endpoint documentation not found"
    
    for line in lines:
        if start_marker in line:
            context_start = True
            continue
        
        if context_start:
            if line.startswith('### ') and start_marker not in line:
                break
            context_lines.append(line)
    
    return '\n'.join(context_lines[:50])  # Limit context size

# === BYBIT PUBLIC ENDPOINTS ===

@app.get("/bybit/tickers", response_model=BybitResponse, operation_id="bybit_get_tickers", tags=["Bybit Public"])
async def get_tickers(
    category: str = Query(..., description="Trading category: spot, linear, inverse, option"),
    symbol: Optional[str] = Query(None, description="Specific symbol (e.g., BTCUSDT)")
):
    """
    Get ticker information for trading symbols
    
    API Context:
    - Endpoint: GET /v5/market/tickers
    - Authentication: Not required (Public endpoint)
    - Rate limit: 10 requests per 1 second
    """
    context = get_endpoint_context('market')
    try:
        result = await bybit_client.get_tickers(category=category, symbol=symbol)
        return BybitResponse(
            success=True,
            data=result,
            api_context=f"Endpoint: GET /v5/market/tickers\\n{context[:200]}..."
        )
    except Exception as e:
        return BybitResponse(
            success=False,
            error=str(e),
            api_context=context[:200]
        )

@app.get("/bybit/kline", response_model=BybitResponse, operation_id="bybit_get_kline", tags=["Bybit Public"])
async def get_kline(
    category: str = Query(..., description="Trading category: spot, linear, inverse, option"),
    symbol: str = Query(..., description="Symbol (e.g., BTCUSDT)"),
    interval: str = Query("1", description="Kline interval: 1,3,5,15,30,60,120,240,360,720,D,W,M"),
    limit: int = Query(5, description="Number of results (1-1000)", ge=1, le=1000)
):
    """
    Get kline/candlestick data
    
    API Context:
    - Endpoint: GET /v5/market/kline
    - Authentication: Not required (Public endpoint)
    - Rate limit: 10 requests per 1 second
    """
    context = get_endpoint_context('market')
    try:
        result = await bybit_client.get_kline(
            category=category, 
            symbol=symbol, 
            interval=interval, 
            limit=limit
        )
        return BybitResponse(
            success=True,
            data=result,
            api_context=f"Endpoint: GET /v5/market/kline\\n{context[:200]}..."
        )
    except Exception as e:
        return BybitResponse(
            success=False,
            error=str(e),
            api_context=context[:200]
        )

@app.get("/bybit/orderbook", response_model=BybitResponse, operation_id="bybit_get_orderbook", tags=["Bybit Public"])
async def get_orderbook(
    category: str = Query(..., description="Trading category: spot, linear, inverse, option"),
    symbol: str = Query(..., description="Symbol (e.g., BTCUSDT)"),
    limit: int = Query(10, description="Number of price levels (1-500)", ge=1, le=500)
):
    """
    Get order book data
    
    API Context: 
    - Endpoint: GET /v5/market/orderbook
    - Authentication: Not required (Public endpoint)
    - Rate limit: 10 requests per 1 second
    """
    context = get_endpoint_context('market')
    try:
        result = await bybit_client.get_orderbook(
            category=category,
            symbol=symbol,
            limit=limit
        )
        return BybitResponse(
            success=True,
            data=result,
            api_context=f"Endpoint: GET /v5/market/orderbook\\n{context[:200]}..."
        )
    except Exception as e:
        return BybitResponse(
            success=False,
            error=str(e),
            api_context=context[:200]
        )

@app.get("/bybit/recent-trades", response_model=BybitResponse, operation_id="bybit_get_recent_trades", tags=["Bybit Public"])
async def get_recent_trades(
    category: str = Query(..., description="Trading category: spot, linear, inverse, option"),
    symbol: str = Query(..., description="Symbol (e.g., BTCUSDT)"),
    limit: int = Query(5, description="Number of trades (1-1000)", ge=1, le=1000)
):
    """
    Get recent trade data
    
    API Context:
    - Endpoint: GET /v5/market/recent-trade
    - Authentication: Not required (Public endpoint)
    - Rate limit: 10 requests per 1 second
    """
    context = get_endpoint_context('market')
    try:
        result = await bybit_client.get_recent_trades(
            category=category,
            symbol=symbol,
            limit=limit
        )
        return BybitResponse(
            success=True,
            data=result,
            api_context=f"Endpoint: GET /v5/market/recent-trade\\n{context[:200]}..."
        )
    except Exception as e:
        return BybitResponse(
            success=False,
            error=str(e),
            api_context=context[:200]
        )

@app.get("/bybit/instruments-info", response_model=BybitResponse, operation_id="bybit_get_instruments_info", tags=["Bybit Public"])
async def get_instruments_info(
    category: str = Query(..., description="Trading category: spot, linear, inverse, option"),
    symbol: Optional[str] = Query(None, description="Specific symbol (e.g., BTCUSDT)")
):
    """
    Get instrument information
    
    API Context:
    - Endpoint: GET /v5/market/instruments-info
    - Authentication: Not required (Public endpoint)
    - Rate limit: 10 requests per 1 second
    """
    context = get_endpoint_context('market')
    try:
        result = await bybit_client.get_instruments_info(
            category=category,
            symbol=symbol
        )
        return BybitResponse(
            success=True,
            data=result,
            api_context=f"Endpoint: GET /v5/market/instruments-info\\n{context[:200]}..."
        )
    except Exception as e:
        return BybitResponse(
            success=False,
            error=str(e),
            api_context=context[:200]
        )

@app.get("/bybit/server-time", response_model=BybitResponse, operation_id="bybit_get_server_time", tags=["Bybit Public"])
async def get_server_time():
    """
    Get Bybit server time
    
    API Context:
    - Endpoint: GET /v5/market/time
    - Authentication: Not required (Public endpoint)
    - Rate limit: 10 requests per 1 second
    """
    context = get_endpoint_context('market')
    try:
        result = await bybit_client.get_server_time()
        return BybitResponse(
            success=True,
            data=result,
            api_context=f"Endpoint: GET /v5/market/time\\n{context[:200]}..."
        )
    except Exception as e:
        return BybitResponse(
            success=False,
            error=str(e),
            api_context=context[:200]
        )

# === BYBIT PRIVATE ENDPOINTS ===

@app.get("/bybit/wallet-balance", response_model=BybitResponse, operation_id="bybit_get_wallet_balance", tags=["Bybit Private"])
async def get_wallet_balance(
    account_type: str = Query("UNIFIED", description="Account type: UNIFIED, SPOT, CONTRACT, INVESTMENT, OPTION")
):
    """
    Get wallet balance (requires API key)
    
    API Context:
    - Endpoint: GET /v5/account/wallet-balance
    - Authentication: Required (Private endpoint)
    - Rate limit: 10 requests per 1 second
    """
    if not bybit_client.can_access_private_endpoints():
        return BybitResponse(
            success=False,
            error="🔒 Private endpoint access requires API credentials and public_only=false",
            api_context=get_endpoint_context('account')[:200]
        )
    
    context = get_endpoint_context('account')
    try:
        result = await bybit_client.get_wallet_balance(account_type=account_type)
        return BybitResponse(
            success=True,
            data=result,
            api_context=f"Endpoint: GET /v5/account/wallet-balance\\n{context[:200]}..."
        )
    except Exception as e:
        return BybitResponse(
            success=False,
            error=str(e),
            api_context=context[:200]
        )

@app.get("/bybit/positions", response_model=BybitResponse, operation_id="bybit_get_positions", tags=["Bybit Private"])
async def get_positions(
    category: str = Query(..., description="Position category: linear, inverse, option"),
    symbol: Optional[str] = Query(None, description="Specific symbol (e.g., BTCUSDT)")
):
    """
    Get position list (requires API key)
    
    API Context:
    - Endpoint: GET /v5/position/list
    - Authentication: Required (Private endpoint)
    - Rate limit: 10 requests per 1 second
    """
    if not bybit_client.can_access_private_endpoints():
        return BybitResponse(
            success=False,
            error="🔒 Private endpoint access requires API credentials and public_only=false",
            api_context=get_endpoint_context('position')[:200]
        )
    
    context = get_endpoint_context('position')
    try:
        result = await bybit_client.get_position_list(
            category=category,
            symbol=symbol
        )
        return BybitResponse(
            success=True,
            data=result,
            api_context=f"Endpoint: GET /v5/position/list\\n{context[:200]}..."
        )
    except Exception as e:
        return BybitResponse(
            success=False,
            error=str(e),
            api_context=context[:200]
        )

# Trading endpoints
class OrderRequest(BaseModel):
    category: str = Field(..., description="Trading category: spot, linear, inverse, option")
    symbol: str = Field(..., description="Symbol (e.g., BTCUSDT)")
    side: str = Field(..., description="Order side: Buy, Sell")
    order_type: str = Field(..., description="Order type: Market, Limit")
    qty: str = Field(..., description="Order quantity")
    price: Optional[str] = Field(None, description="Order price (required for Limit orders)")

@app.post("/bybit/order", response_model=BybitResponse, operation_id="bybit_create_order", tags=["Bybit Private"])
async def create_order(order: OrderRequest):
    """
    Create new order (requires API key)
    
    API Context:
    - Endpoint: POST /v5/order/create
    - Authentication: Required (Private endpoint)
    - Rate limit: 10 requests per 1 second
    """
    if not bybit_client.can_access_private_endpoints():
        return BybitResponse(
            success=False,
            error="🔒 Trading requires API credentials and public_only=false",
            api_context=get_endpoint_context('order')[:200]
        )
    
    context = get_endpoint_context('order')
    try:
        result = await bybit_client.create_order(
            category=order.category,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            qty=order.qty,
            price=order.price
        )
        return BybitResponse(
            success=True,
            data=result,
            api_context=f"Endpoint: POST /v5/order/create\\n{context[:200]}..."
        )
    except Exception as e:
        return BybitResponse(
            success=False,
            error=str(e),
            api_context=context[:200]
        )

# === API DOCUMENTATION ENDPOINT ===

@app.get("/api-docs", response_model=APIDocsResponse, operation_id="get_api_docs_context", tags=["Documentation"])
async def get_api_docs_context(
    section: Optional[str] = Query(None, description="Specific section: market, order, position, account, asset")
):
    """
    Get Bybit V5 API documentation context
    
    Provides comprehensive API documentation for specific sections or full context
    """
    try:
        if section:
            context = get_endpoint_context(section)
            if "not found" in context:
                return APIDocsResponse(
                    success=False,
                    data={"error": f"Section '{section}' not found. Available: market, order, position, account, asset"}
                )
        else:
            context = api_docs_context[:2000]  # Limit full context
        
        return APIDocsResponse(
            success=True,
            data={
                "documentation": context,
                "total_length": len(api_docs_context),
                "excerpt_length": len(context)
            },
            section=section or "full",
            available_sections=["market", "order", "position", "account", "asset"]
        )
    except Exception as e:
        return APIDocsResponse(
            success=False,
            data={"error": str(e)}
        )

# === MULTI-EXCHANGE ENDPOINTS ===

@app.get("/exchanges/binance/ticker", response_model=Dict[str, Any], operation_id="binance_get_ticker", tags=["Multi-Exchange"])
async def binance_get_ticker(symbol: Optional[str] = Query(None, description="Symbol (e.g., BTCUSDT)")):
    """
    Get Binance ticker information
    
    API Context:
    - Exchange: Binance
    - Endpoint: GET /api/v3/ticker/price or /api/v3/ticker/24hr
    - Authentication: Not required (Public endpoint)
    """
    try:
        import httpx
        base_url = "https://api.binance.com/api/v3"
        
        if symbol:
            # Get specific symbol ticker
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{base_url}/ticker/24hr?symbol={symbol}")
                result = response.json()
        else:
            # Get all tickers
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{base_url}/ticker/24hr")
                result = response.json()
        
        return {
            "success": True,
            "exchange": "binance",
            "data": result,
            "api_context": f"Binance API: GET /api/v3/ticker/24hr - Public endpoint"
        }
    except Exception as e:
        return {
            "success": False,
            "exchange": "binance",
            "error": str(e)
        }

@app.get("/exchanges/kucoin/ticker", response_model=Dict[str, Any], operation_id="kucoin_get_ticker", tags=["Multi-Exchange"])
async def kucoin_get_ticker(symbol: Optional[str] = Query(None, description="Symbol (e.g., BTC-USDT)")):
    """
    Get KuCoin ticker information
    
    API Context:
    - Exchange: KuCoin
    - Endpoint: GET /api/v1/market/allTickers or /api/v1/market/stats
    - Authentication: Not required (Public endpoint)
    """
    try:
        import httpx
        base_url = "https://api.kucoin.com"
        
        if symbol:
            # Get specific symbol ticker
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{base_url}/api/v1/market/stats?symbol={symbol}")
                result = response.json()
        else:
            # Get all tickers
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{base_url}/api/v1/market/allTickers")
                result = response.json()
        
        return {
            "success": True,
            "exchange": "kucoin",
            "data": result,
            "api_context": f"KuCoin API: GET /api/v1/market/allTickers - Public endpoint"
        }
    except Exception as e:
        return {
            "success": False,
            "exchange": "kucoin",
            "error": str(e)
        }

@app.get("/exchanges/okx/ticker", response_model=Dict[str, Any], operation_id="okx_get_ticker", tags=["Multi-Exchange"])
async def okx_get_ticker(symbol: Optional[str] = Query(None, description="Symbol (e.g., BTC-USDT)")):
    """
    Get OKX ticker information
    
    API Context:
    - Exchange: OKX
    - Endpoint: GET /api/v5/market/ticker or /api/v5/market/tickers
    - Authentication: Not required (Public endpoint)
    """
    try:
        import httpx
        base_url = "https://www.okx.com/api/v5"
        
        if symbol:
            # Get specific symbol ticker
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{base_url}/market/ticker?instId={symbol}")
                result = response.json()
        else:
            # Get all spot tickers
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{base_url}/market/tickers?instType=SPOT")
                result = response.json()
        
        return {
            "success": True,
            "exchange": "okx",
            "data": result,
            "api_context": f"OKX API: GET /api/v5/market/ticker - Public endpoint"
        }
    except Exception as e:
        return {
            "success": False,
            "exchange": "okx",
            "error": str(e)
        }

@app.get("/exchanges/huobi/ticker", response_model=Dict[str, Any], operation_id="huobi_get_ticker", tags=["Multi-Exchange"])
async def huobi_get_ticker(symbol: Optional[str] = Query(None, description="Symbol (e.g., btcusdt)")):
    """
    Get Huobi ticker information
    
    API Context:
    - Exchange: Huobi
    - Endpoint: GET /market/detail/merged or /market/tickers
    - Authentication: Not required (Public endpoint)
    """
    try:
        import httpx
        base_url = "https://api.huobi.pro"
        
        if symbol:
            # Get specific symbol ticker
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{base_url}/market/detail/merged?symbol={symbol.lower()}")
                result = response.json()
        else:
            # Get all tickers
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{base_url}/market/tickers")
                result = response.json()
        
        return {
            "success": True,
            "exchange": "huobi",
            "data": result,
            "api_context": f"Huobi API: GET /market/tickers - Public endpoint"
        }
    except Exception as e:
        return {
            "success": False,
            "exchange": "huobi",
            "error": str(e)
        }

@app.get("/exchanges/mexc/ticker", response_model=Dict[str, Any], operation_id="mexc_get_ticker", tags=["Multi-Exchange"])
async def mexc_get_ticker(symbol: Optional[str] = Query(None, description="Symbol (e.g., BTCUSDT)")):
    """
    Get MEXC ticker information
    
    API Context:
    - Exchange: MEXC
    - Endpoint: GET /api/v3/ticker/price or /api/v3/ticker/24hr
    - Authentication: Not required (Public endpoint)
    """
    try:
        import httpx
        base_url = "https://api.mexc.com/api/v3"
        
        if symbol:
            # Get specific symbol ticker
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{base_url}/ticker/24hr?symbol={symbol}")
                result = response.json()
        else:
            # Get all tickers
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{base_url}/ticker/24hr")
                result = response.json()
        
        return {
            "success": True,
            "exchange": "mexc",
            "data": result,
            "api_context": f"MEXC API: GET /api/v3/ticker/24hr - Public endpoint"
        }
    except Exception as e:
        return {
            "success": False,
            "exchange": "mexc",
            "error": str(e)
        }

# === UNIFIED MULTI-EXCHANGE ENDPOINT ===

@app.get("/exchanges/compare-prices", response_model=Dict[str, Any], operation_id="compare_exchange_prices", tags=["Multi-Exchange"])
async def compare_exchange_prices(
    symbol: str = Query(..., description="Symbol to compare (format varies by exchange)"),
    exchanges: str = Query("bybit,binance,kucoin", description="Comma-separated list of exchanges")
):
    """
    Compare prices across multiple exchanges
    
    This endpoint fetches ticker data from multiple exchanges simultaneously
    and provides price comparison with spread analysis
    """
    try:
        import httpx
        import asyncio
        
        exchange_list = [ex.strip() for ex in exchanges.split(",")]
        results = {}
        
        async def fetch_bybit():
            try:
                result = await bybit_client.get_tickers(category="spot", symbol=symbol)
                if result.get("result", {}).get("list"):
                    ticker = result["result"]["list"][0]
                    return {
                        "price": float(ticker.get("lastPrice", 0)),
                        "volume24h": float(ticker.get("volume24h", 0)),
                        "change24h": float(ticker.get("price24hPcnt", 0)) * 100
                    }
            except:
                pass
            return None
        
        async def fetch_binance():
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}")
                    data = response.json()
                    return {
                        "price": float(data.get("lastPrice", 0)),
                        "volume24h": float(data.get("volume", 0)),
                        "change24h": float(data.get("priceChangePercent", 0))
                    }
            except:
                pass
            return None
        
        async def fetch_kucoin():
            try:
                # Convert BTCUSDT to BTC-USDT format
                kucoin_symbol = symbol.replace("USDT", "-USDT").replace("BTC", "BTC")
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"https://api.kucoin.com/api/v1/market/stats?symbol={kucoin_symbol}")
                    data = response.json()
                    if data.get("data"):
                        return {
                            "price": float(data["data"].get("last", 0)),
                            "volume24h": float(data["data"].get("vol", 0)),
                            "change24h": float(data["data"].get("changeRate", 0)) * 100
                        }
            except:
                pass
            return None
        
        # Fetch data from requested exchanges
        tasks = []
        if "bybit" in exchange_list:
            tasks.append(("bybit", fetch_bybit()))
        if "binance" in exchange_list:
            tasks.append(("binance", fetch_binance()))
        if "kucoin" in exchange_list:
            tasks.append(("kucoin", fetch_kucoin()))
        
        # Execute all requests concurrently
        for exchange_name, task in tasks:
            try:
                result = await task
                if result:
                    results[exchange_name] = result
            except Exception as e:
                results[exchange_name] = {"error": str(e)}
        
        # Calculate spread analysis
        prices = [data["price"] for data in results.values() if isinstance(data, dict) and "price" in data and data["price"] > 0]
        
        analysis = {}
        if len(prices) > 1:
            min_price = min(prices)
            max_price = max(prices)
            spread = max_price - min_price
            spread_percent = (spread / min_price) * 100
            
            analysis = {
                "min_price": min_price,
                "max_price": max_price,
                "spread": spread,
                "spread_percent": round(spread_percent, 4),
                "avg_price": round(sum(prices) / len(prices), 6)
            }
        
        return {
            "success": True,
            "symbol": symbol,
            "exchanges": results,
            "analysis": analysis,
            "timestamp": asyncio.get_event_loop().time()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "symbol": symbol
        }

# === SETUP MCP SERVER ===

# Create MCP server dari FastAPI app
mcp = FastApiMCP(
    app,
    name="Multi-Exchange Trading MCP Server",
    description="MCP Server for Bybit V5 API + Multi-Exchange integration dengan konteks dokumentasi lengkap",
    describe_all_responses=True,
    describe_full_response_schema=True
)

# Mount MCP server ke FastAPI app dengan HTTP transport
mcp.mount_http()

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Bybit MCP Server...")
    print(f"🔧 Public only mode: {config.public_only}")
    print(f"📚 API docs loaded: {len(api_docs_context)} characters")
    print(f"🌐 MCP Server will be available at: http://localhost:8001/mcp")
    print(f"📖 FastAPI docs available at: http://localhost:8001/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)