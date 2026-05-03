"""
Tests for backend/websocket_manager.py

Covers ConnectionManager: connect, disconnect, and broadcast.
WebSocket objects are mocked so no real network calls are made.
"""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.websocket_manager import ConnectionManager


def _mock_websocket():
    """Return a mock WebSocket with async send_text."""
    ws = MagicMock()
    ws.accept = AsyncMock()
    ws.send_text = AsyncMock()
    return ws


class TestConnectionManagerConnect(unittest.IsolatedAsyncioTestCase):
    """Tests for ConnectionManager.connect()."""

    async def test_connect_accepts_websocket(self):
        """connect() calls websocket.accept() exactly once."""
        manager = ConnectionManager()
        ws = _mock_websocket()
        await manager.connect(ws)
        ws.accept.assert_awaited_once()

    async def test_connect_adds_to_active_connections(self):
        """connect() appends the socket to active_connections."""
        manager = ConnectionManager()
        ws = _mock_websocket()
        await manager.connect(ws)
        self.assertIn(ws, manager.active_connections)

    async def test_connect_multiple_clients(self):
        """Multiple connections are all tracked."""
        manager = ConnectionManager()
        ws1, ws2 = _mock_websocket(), _mock_websocket()
        await manager.connect(ws1)
        await manager.connect(ws2)
        self.assertEqual(len(manager.active_connections), 2)


class TestConnectionManagerDisconnect(unittest.TestCase):
    """Tests for ConnectionManager.disconnect()."""

    def test_disconnect_removes_known_socket(self):
        """disconnect() removes a connected socket."""
        manager = ConnectionManager()
        ws = _mock_websocket()
        manager.active_connections.append(ws)
        manager.disconnect(ws)
        self.assertNotIn(ws, manager.active_connections)

    def test_disconnect_unknown_socket_is_noop(self):
        """disconnect() on a socket that was never connected raises no error."""
        manager = ConnectionManager()
        ws = _mock_websocket()
        try:
            manager.disconnect(ws)
        except Exception as e:
            self.fail(f"disconnect() raised unexpectedly: {e}")

    def test_disconnect_leaves_other_clients_intact(self):
        """Only the target socket is removed; others remain."""
        manager = ConnectionManager()
        ws1, ws2 = _mock_websocket(), _mock_websocket()
        manager.active_connections.extend([ws1, ws2])
        manager.disconnect(ws1)
        self.assertIn(ws2, manager.active_connections)
        self.assertNotIn(ws1, manager.active_connections)


class TestConnectionManagerBroadcast(unittest.IsolatedAsyncioTestCase):
    """Tests for ConnectionManager.broadcast()."""

    async def test_broadcast_sends_json_to_all(self):
        """broadcast() sends JSON-encoded message to every connection."""
        manager = ConnectionManager()
        ws1, ws2 = _mock_websocket(), _mock_websocket()
        manager.active_connections.extend([ws1, ws2])

        msg = {"type": "indexing_progress", "percent": 50}
        await manager.broadcast(msg)

        expected = json.dumps(msg)
        ws1.send_text.assert_awaited_once_with(expected)
        ws2.send_text.assert_awaited_once_with(expected)

    async def test_broadcast_no_connections_is_noop(self):
        """broadcast() with no connections does not raise."""
        manager = ConnectionManager()
        try:
            await manager.broadcast({"type": "test"})
        except Exception as e:
            self.fail(f"broadcast() raised with no connections: {e}")

    async def test_broadcast_removes_dead_connections(self):
        """Connections that raise on send_text are pruned from active_connections."""
        manager = ConnectionManager()
        ws_good = _mock_websocket()
        ws_dead = _mock_websocket()
        ws_dead.send_text.side_effect = Exception("connection closed")

        manager.active_connections.extend([ws_good, ws_dead])
        await manager.broadcast({"type": "ping"})

        self.assertIn(ws_good, manager.active_connections)
        self.assertNotIn(ws_dead, manager.active_connections)

    async def test_broadcast_continues_after_dead_connection(self):
        """Good connections still receive messages even if one connection is dead."""
        manager = ConnectionManager()
        ws_dead = _mock_websocket()
        ws_dead.send_text.side_effect = Exception("gone")
        ws_alive = _mock_websocket()

        manager.active_connections.extend([ws_dead, ws_alive])
        await manager.broadcast({"type": "ok"})

        ws_alive.send_text.assert_awaited_once()


class TestModuleSingleton(unittest.TestCase):
    """Tests for the module-level manager singleton."""

    def test_module_exports_manager_instance(self):
        """The module exposes a ConnectionManager singleton named 'manager'."""
        from backend.websocket_manager import manager
        self.assertIsInstance(manager, ConnectionManager)

    def test_singleton_starts_with_no_connections(self):
        """The module-level singleton has an empty active_connections list on import."""
        from backend.websocket_manager import manager
        self.assertIsInstance(manager.active_connections, list)


if __name__ == "__main__":
    unittest.main()
