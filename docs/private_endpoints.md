Private Endpoints Overview

This document will collect notes and examples for Bybit V5 private endpoints used by the bot (requires valid API keys and non-public-only mode).

Scope

- Wallet balance: `GET /v5/account/wallet-balance`
- Positions: `GET /v5/position/list`
- Orders: create/cancel endpoints (future work)

Notes

- Set `BYBIT_PUBLIC_ONLY=false` and provide `BYBIT_API_KEY` and `BYBIT_API_SECRET` in `.env` to enable private endpoints.
- When credentials are missing or `public_only=true`, the bot gracefully informs the user instead of making private calls.
