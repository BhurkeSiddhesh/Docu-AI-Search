"""
backend/system_prompts.py
-------------------------
Database-backed system prompt library for reusable personas and instructions.

Provides CRUD operations for system prompts stored in SQLite, plus a set of
built-in default prompts that are seeded on first run.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from backend import database

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default system prompts (seeded on first run)
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS = [
    {
        "name": "Document Analyst",
        "content": (
            "You are a precise document analysis assistant. "
            "Answer ONLY using facts from the provided documents. "
            "Quote specific data, numbers, and details. "
            "Reference file names when possible. "
            "If the documents don't contain relevant information, state that clearly."
        ),
        "category": "analysis",
    },
    {
        "name": "Code Reviewer",
        "content": (
            "You are a senior software engineer reviewing code found in documents. "
            "Focus on: correctness, potential bugs, security issues, and best practices. "
            "Provide actionable suggestions with code examples where relevant."
        ),
        "category": "development",
    },
    {
        "name": "Creative Summariser",
        "content": (
            "You are a creative writer who produces engaging summaries. "
            "Transform dry document content into easy-to-read narratives. "
            "Use vivid language but stay faithful to the source material. "
            "Structure your response with clear headings and bullet points."
        ),
        "category": "creative",
    },
    {
        "name": "Research Assistant",
        "content": (
            "You are a thorough research assistant. "
            "Cross-reference information across multiple documents when available. "
            "Highlight contradictions, gaps, and areas needing further investigation. "
            "Organise findings into a structured format with citations."
        ),
        "category": "research",
    },
    {
        "name": "ELI5 Explainer",
        "content": (
            "You explain complex document content in simple terms that anyone can understand. "
            "Use analogies and everyday examples. Avoid jargon. "
            "If technical terms must be used, define them immediately."
        ),
        "category": "general",
    },
]


# ---------------------------------------------------------------------------
# CRUD helpers (all delegate to database.py)
# ---------------------------------------------------------------------------

def seed_default_prompts() -> None:
    """Insert default prompts if the table is empty. Called once on startup."""
    existing = get_system_prompts()
    if existing:
        return
    logger.info("[SystemPrompts] Seeding %d default prompts", len(DEFAULT_PROMPTS))
    for p in DEFAULT_PROMPTS:
        add_system_prompt(p["name"], p["content"], p["category"])


def add_system_prompt(name: str, content: str, category: str = "general") -> int:
    """Insert a new system prompt. Returns the new row ID."""
    conn = database.get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO system_prompts (name, content, category) VALUES (?, ?, ?)",
        (name.strip(), content.strip(), category.strip()),
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def get_system_prompts(category: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return all system prompts, optionally filtered by category."""
    conn = database.get_connection()
    cursor = conn.cursor()
    if category:
        cursor.execute(
            "SELECT id, name, content, category, created_at FROM system_prompts WHERE category = ? ORDER BY name",
            (category,),
        )
    else:
        cursor.execute(
            "SELECT id, name, content, category, created_at FROM system_prompts ORDER BY name"
        )
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "id": r[0],
            "name": r[1],
            "content": r[2],
            "category": r[3],
            "created_at": r[4],
        }
        for r in rows
    ]


def get_system_prompt_by_id(prompt_id: int) -> Optional[Dict[str, Any]]:
    """Return a single prompt by ID, or ``None``."""
    conn = database.get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, name, content, category, created_at FROM system_prompts WHERE id = ?",
        (prompt_id,),
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row[0],
        "name": row[1],
        "content": row[2],
        "category": row[3],
        "created_at": row[4],
    }


def update_system_prompt(prompt_id: int, name: str, content: str, category: str = "general") -> bool:
    """Update an existing prompt. Returns True on success."""
    conn = database.get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE system_prompts SET name = ?, content = ?, category = ? WHERE id = ?",
        (name.strip(), content.strip(), category.strip(), prompt_id),
    )
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    return affected > 0


def delete_system_prompt(prompt_id: int) -> bool:
    """Delete a prompt by ID. Returns True if a row was removed."""
    conn = database.get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM system_prompts WHERE id = ?", (prompt_id,))
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    return affected > 0
