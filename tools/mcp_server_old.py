#!/usr/bin/env python3
"""
Model Context Protocol (MCP) Server untuk Bybit API
Server ini menyediakan tools untuk mengakses Bybit API secara dinamis
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Sequence
from datetime import datetime

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from bybit_client import BybitClient, BybitConfig
from config import get_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPBybitServer:
    """MCP Server untuk Bybit API integration"""
    
    def __init__(self):
        self.server = Server("bybit-mcp")
        self.config = get_config()
        self.bybit_client = BybitClient(BybitConfig(
            api_key=self.config.bybit_api_key,
            api_secret=self.config.bybit_api_secret,
            testnet=self.config.bybit_testnet,
            public_only=self.config.public_only
        ))
        self._register_tools()

    def _register_tools(self):
        """Register all MCP tools"""
        
        @self.server.list_tools()
        async def list_tools() -> List[types.Tool]:
            """List semua tools yang tersedia"""
            public_tools = [
                types.Tool(
                    name="get_tickers",
                    description="Get ticker information for trading symbols",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": ["spot", "linear", "inverse", "option"],
                                "description": "Trading category"
                            },
                            "symbol": {
                                "type": "string",
                                "description": "Symbol (optional, e.g., BTCUSDT)"
                            }
                        },
                        "required": ["category"]
                    }
                ),
                types.Tool(
                    name="get_kline",
                    description="Get kline/candlestick data",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": ["spot", "linear", "inverse", "option"],
                                "description": "Trading category"
                            },
                            "symbol": {
                                "type": "string",
                                "description": "Symbol (e.g., BTCUSDT)"
                            },
                            "interval": {
                                "type": "string",
                                "enum": ["1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"],
                                "description": "Kline interval",
                                "default": "1"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Number of results (1-1000)",
                                "minimum": 1,
                                "maximum": 1000,
                                "default": 5
                            }
                        },
                        "required": ["category", "symbol"]
                    }
                ),
                types.Tool(
                    name="get_orderbook",
                    description="Get order book data",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": ["spot", "linear", "inverse", "option"],
                                "description": "Trading category"
                            },
                            "symbol": {
                                "type": "string",
                                "description": "Symbol (e.g., BTCUSDT)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Number of price levels (1-500)",
                                "minimum": 1,
                                "maximum": 500,
                                "default": 10
                            }
                        },
                        "required": ["category", "symbol"]
                    }
                ),
                types.Tool(
                    name="get_recent_trades",
                    description="Get recent trade data",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": ["spot", "linear", "inverse", "option"],
                                "description": "Trading category"
                            },
                            "symbol": {
                                "type": "string",
                                "description": "Symbol (e.g., BTCUSDT)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Number of trades (1-1000)",
                                "minimum": 1,
                                "maximum": 1000,
                                "default": 5
                            }
                        },
                        "required": ["category", "symbol"]
                    }
                ),
                types.Tool(
                    name="get_instruments_info",
                    description="Get instrument information",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": ["spot", "linear", "inverse", "option"],
                                "description": "Trading category"
                            },
                            "symbol": {
                                "type": "string",
                                "description": "Symbol (optional, e.g., BTCUSDT)"
                            }
                        },
                        "required": ["category"]
                    }
                ),
                types.Tool(
                    name="get_server_time",
                    description="Get Bybit server time",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]
            
            private_tools = [
                types.Tool(
                    name="get_wallet_balance",
                    description="Get wallet balance (requires API key)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "accountType": {
                                "type": "string",
                                "enum": ["UNIFIED", "SPOT", "CONTRACT", "INVESTMENT", "OPTION"],
                                "description": "Account type",
                                "default": "UNIFIED"
                            }
                        }
                    }
                ),
                types.Tool(
                    name="get_positions",
                    description="Get position list (requires API key)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "enum": ["linear", "inverse", "option"],
                                "description": "Position category"
                            },
                            "symbol": {
                                "type": "string",
                                "description": "Symbol (optional, e.g., BTCUSDT)"
                            }
                        },
                        "required": ["category"]
                    }
                )
            ]
            
            # Return only public tools if public_only is True
            if self.config.public_only:
                return public_tools
            else:
                return public_tools + private_tools

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> Sequence[types.TextContent]:
            """Execute tool berdasarkan nama dan arguments"""
            
            try:
                if name == "get_tickers":
                    result = await self.bybit_client.get_tickers(
                        category=arguments["category"],
                        symbol=arguments.get("symbol")
                    )
                    
                elif name == "get_kline":
                    result = await self.bybit_client.get_kline(
                        category=arguments["category"],
                        symbol=arguments["symbol"],
                        interval=arguments.get("interval", "1"),
                        limit=arguments.get("limit", 5)
                    )
                    
                elif name == "get_orderbook":
                    result = await self.bybit_client.get_orderbook(
                        category=arguments["category"],
                        symbol=arguments["symbol"],
                        limit=arguments.get("limit", 10)
                    )
                    
                elif name == "get_recent_trades":
                    result = await self.bybit_client.get_recent_trades(
                        category=arguments["category"],
                        symbol=arguments["symbol"],
                        limit=arguments.get("limit", 5)
                    )
                    
                elif name == "get_instruments_info":
                    result = await self.bybit_client.get_instruments_info(
                        category=arguments["category"],
                        symbol=arguments.get("symbol")
                    )
                    
                elif name == "get_wallet_balance":
                    if not self.bybit_client.can_access_private_endpoints():
                        return [types.TextContent(
                            type="text",
                            text="Error: Wallet balance requires API key configuration and public_only mode to be disabled"
                        )]
                    result = await self.bybit_client.get_wallet_balance(
                        account_type=arguments.get("accountType", "UNIFIED")
                    )
                    
                elif name == "get_positions":
                    if not self.bybit_client.can_access_private_endpoints():
                        return [types.TextContent(
                            type="text",
                            text="Error: Position data requires API key configuration and public_only mode to be disabled"
                        )]
                    result = await self.bybit_client.get_position_list(
                        category=arguments["category"],
                        symbol=arguments.get("symbol")
                    )
                    
                elif name == "get_server_time":
                    result = await self.bybit_client.get_server_time()
                    
                else:
                    return [types.TextContent(
                        type="text",
                        text=f"Error: Unknown tool '{name}'"
                    )]
                
                # Return successful result
                return [types.TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]
                
            except Exception as e:
                logger.error(f"Error executing tool {name}: {e}")
                return [types.TextContent(
                    type="text",
                    text=f"Error executing {name}: {str(e)}"
                )]

    async def run(self):
        """Run MCP server"""
        logger.info("Starting MCP Bybit Server...")
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )

async def main():
    """Main function"""
    mcp_server = MCPBybitServer()
    await mcp_server.run()

if __name__ == "__main__":
    asyncio.run(main())
