"""
WebSocket connection manager for real-time progress updates.

Clients connect to ws://host/ws/progress and receive JSON events:
  {"type": "indexing_progress", "percent": 42, "current_file": "doc.pdf", "total": 100}
  {"type": "indexing_complete", "total_files": 87, "duration_seconds": 12.4}
  {"type": "download_progress", "model_id": "...", "percent": 60}
  {"type": "error", "message": "..."}
"""

import asyncio
import json
import logging
from typing import List
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections and broadcasts messages."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock: asyncio.Lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info("WebSocket client connected. Total: %d", len(self.active_connections))

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info("WebSocket client disconnected. Total: %d", len(self.active_connections))

    async def broadcast(self, message: dict):
        """Send a JSON message to all connected clients."""
        text = json.dumps(message)
        async with self._lock:
            targets = list(self.active_connections)
        dead = []
        for connection in targets:
            try:
                await connection.send_text(text)
            except Exception:
                dead.append(connection)
        for conn in dead:
            await self.disconnect(conn)


# Module-level singleton shared across the app
manager = ConnectionManager()
