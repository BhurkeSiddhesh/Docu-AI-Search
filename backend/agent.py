import re
import asyncio
from typing import List, Dict, Any, Generator
from backend import tools, llm_integration

# Patterns that indicate a direct natural-language answer.
# Only used AFTER at least one search has been performed.
_DIRECT_ANSWER_PREFIXES = (
    "based on the document",
    "based on the file",
    "based on the search",
    "according to the document",
    "according to the file",
    "according to the search result",
    "from the document",
    "from the file",
    "the document says",
    "the document states",
    "the file says",
    "the file states",
)

# Simplified prompt for local/small models (Gemma-2B, TinyLlama, Phi-2, etc.)
# Key rule: ALWAYS search first — no hallucination allowed.
_LOCAL_SYSTEM_PROMPT = """You are a document search assistant. You can ONLY use information from the search results.

AVAILABLE TOOLS:
- search_knowledge_base: Search the indexed documents
- list_files: List all indexed files
- read_file: Read a specific file

STRICT RULES:
1. ALWAYS use search_knowledge_base FIRST before anything else
2. NEVER make up or use prior knowledge — only use search results
3. If no relevant info is found in search results, say "No information found"

FORMAT (use EXACTLY this, every time):
Thought: I need to search for information about this
Action: search_knowledge_base
Action Input: your search terms here

After you get the Observation, write your answer:
Final Answer: [answer using ONLY information from the Observation]
"""

# Full ReAct prompt for capable cloud models
_CLOUD_SYSTEM_PROMPT = """You are an advanced Research Assistant. Answer the user's question by thinking step-by-step and using tools.

AVAILABLE TOOLS:
- search_knowledge_base(query): Search the indexed documents for concepts, people, and data.
- list_files(): List the names of all indexed files.
- read_file(filename_or_path): Read the full text content of a specific file.

STRICT FORMAT:
Thought: (Explain what you are looking for and why)
Action: (Exactly one of: search_knowledge_base, list_files, read_file)
Action Input: (The parameter for the tool)

Wait for the "Observation:" from the system before continuing.
When you have the final answer, use this format:
Final Answer: (Your detailed response citing sources)

CRITICAL:
- ALWAYS search first. NEVER answer from prior knowledge.
- If searching for a person, use their exact name.
- Do NOT hallucinate info not in the documents.
"""


class ReActAgent:
    def __init__(self, global_state: dict):
        self.global_state = global_state

        config = global_state['config']
        self.provider = config.get('LocalLLM', 'provider', fallback='openai')
        self.api_key = config.get('APIKeys', f"{self.provider}_api_key", fallback='')
        self.model_path = config.get('LocalLLM', 'model_path', fallback='')

        self.is_local = (self.provider == 'local')
        # Local models: fewer steps (prevents multi-search spiral), smaller context
        self.max_steps = 4 if self.is_local else 5
        self.system_prompt = _LOCAL_SYSTEM_PROMPT if self.is_local else _CLOUD_SYSTEM_PROMPT
        self.max_tokens = 384 if self.is_local else 512
        # Max chars of each observation to include in history (keeps context lean)
        self.obs_window = 350 if self.is_local else 800

    def _extract_final_answer(self, text: str) -> str | None:
        """
        Flexibly extracts final answer from many surface forms.
        """
        # Standard form
        if "Final Answer:" in text:
            return text.split("Final Answer:")[-1].strip()
        if "final answer:" in text.lower():
            idx = text.lower().index("final answer:")
            return text[idx + len("final answer:"):].strip()
        # "Answer:" prefix (some models)
        m = re.search(r"^Answer:\s*(.+)", text, re.MULTILINE | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return None

    def _is_grounded_direct_answer(self, text: str) -> bool:
        """
        Returns True ONLY if the response starts with a phrase explicitly
        referencing a document/search result (i.e. it's grounded, not hallucinated).
        This is only called AFTER at least one search has happened.
        """
        if len(text.strip()) < 60:
            return False
        if re.search(r"\baction\s*:", text.lower()):
            return False
        t = text.strip().lower()
        return any(t.startswith(prefix) for prefix in _DIRECT_ANSWER_PREFIXES)

    def _force_search_action(self, user_query: str) -> tuple[str, str]:
        """
        Extracts key search terms from the user's question to build a forced search.
        """
        # Strip common question words to get the content
        cleaned = re.sub(
            r'\b(what|where|when|who|why|how|did|does|is|are|was|were|tell me about|describe)\b',
            '', user_query, flags=re.IGNORECASE
        ).strip()
        # Fall back to full query if too much was stripped
        if len(cleaned) < 5:
            cleaned = user_query
        return "search_knowledge_base", cleaned

    async def stream_chat(self, user_query: str) -> Generator[Dict[str, str], None, None]:
        """
        Runs the ReAct loop and streams events (thoughts, actions, answer).
        Yields dicts like: {"type": "thought", "content": "..."}
        """
        history = [f"Question: {user_query}"]
        has_searched = False  # Guard: only allow non-tool answers AFTER a search

        for step in range(self.max_steps):
            # Trim history for local models to avoid context overflow.
            # Keep the first entry (question) + last 4 entries (recent context).
            if self.is_local and len(history) > 6:
                history = [history[0]] + history[-4:]
            user_content = "\n".join(history) + "\n\nThought:"

            try:
                response_text = llm_integration.generate_ai_answer(
                    context="",
                    question=user_content,
                    provider=self.provider,
                    api_key=self.api_key,
                    model_path=self.model_path,
                    raw=True,
                    system_instruction=self.system_prompt,
                    stop=["Observation:", "Question:"],
                    max_tokens=self.max_tokens,
                    temperature=0.1
                )

                if response_text.startswith("Error"):
                    raise Exception(response_text)

            except Exception as e:
                yield {"type": "error", "content": f"LLM Error: {e}"}
                return

            print(f"\n[AGENT STEP {step+1}] {response_text[:300]}\n")

            current_step_log = f"Thought: {response_text}"
            history.append(current_step_log)

            # ── 1. Check for explicit final answer (only valid after a search) ──
            final_ans = self._extract_final_answer(response_text)
            if final_ans:
                if has_searched:
                    print(f"[AGENT FINAL ANSWER] {final_ans[:200]}")
                    yield {"type": "answer", "content": final_ans}
                    return
                else:
                    # Model is trying to answer without searching — force a search
                    print(f"[AGENT BLOCK] Model tried to answer before searching. Forcing search...")
                    yield {"type": "thought", "content": "I need to search the index before answering."}
                    action, action_input = self._force_search_action(user_query)
                    # Fall through to execute the forced action below
                    action_match_str = action
                    action_input_str = action_input
                    # Skip the regex parse, jump straight to tool execution
                    pass
            else:
                action_match_str = None
                action_input_str = None

                # ── 2. Parse Action / Action Input from response ─────────────────
                action_match = re.search(
                    r"Action:\s*(search_knowledge_base|list_files|read_file)",
                    response_text, re.IGNORECASE
                )
                input_match = re.search(
                    r"Action Input:\s*(.+)", response_text, re.IGNORECASE | re.DOTALL
                )

                # Fallback: tool-call style — search_knowledge_base("input")
                if not action_match or not input_match:
                    fallback_match = re.search(
                        r"(search_knowledge_base|list_files|read_file)\s*\(([\"\']?)([^)\"\']+)([\"\']?)\)",
                        response_text, re.IGNORECASE
                    )
                    if fallback_match:
                        action_match_str = fallback_match.group(1).lower()
                        action_input_str = fallback_match.group(3).strip()
                    else:
                        # ── 3. No action found ────────────────────────────────────
                        thought_text = response_text.strip()

                        # If we have NOT searched yet, FORCE a search regardless
                        if not has_searched:
                            print(f"[AGENT FORCE SEARCH step={step}] No action on first step, forcing search")
                            yield {"type": "thought", "content": "Searching the index for relevant information..."}
                            action_match_str, action_input_str = self._force_search_action(user_query)
                            # Fall through to tool execution below

                        # If we HAVE searched and response is document-grounded, accept it
                        elif self._is_grounded_direct_answer(thought_text):
                            print(f"[AGENT GROUNDED ANSWER step={step}] {thought_text[:200]}")
                            yield {"type": "answer", "content": thought_text}
                            return

                        # If we've searched and it's a longish response, take it as answer
                        elif len(thought_text) > 100 and step >= 2:
                            print(f"[AGENT AUTO-ANSWER step={step}] {thought_text[:200]}")
                            yield {"type": "answer", "content": thought_text}
                            return

                        else:
                            if thought_text:
                                yield {"type": "thought", "content": thought_text}
                            nudge = (
                                "Observation: You provided a thought but no Action. "
                                "Please use search_knowledge_base to find information, "
                                "or write 'Final Answer: <your answer>'."
                            )
                            history.append(nudge)
                            continue
                else:
                    action_match_str = action_match.group(1).strip().lower()
                    action_input_str = input_match.group(1).strip().split("\n")[0].strip()

            # ── 4. Execute Tool ──────────────────────────────────────────────────
            action = action_match_str
            action_input = action_input_str

            print(f"[AGENT ACTION] {action}('{action_input}')")
            yield {"type": "thought", "content": f"Searching for: {action_input!r}" if action == "search_knowledge_base" else f"Using tool: {action}"}
            yield {"type": "action", "content": f"Executing {action}..."}

            tool_func = tools.AVAILABLE_TOOLS.get(action)
            if tool_func:
                try:
                    if action == "search_knowledge_base":
                        observation = tool_func(action_input, self.global_state)
                        has_searched = True  # Mark that at least one search happened
                        # For local models: after first search, if observation is rich enough,
                        # nudge the model to answer immediately (avoids multi-search spiral)
                        if self.is_local and len(observation) > 100:
                            observation += "\n\n[SYSTEM: You have found relevant information. Write your Final Answer now based on this Observation.]"
                    else:
                        observation = tool_func(action_input)
                except Exception as e:
                    observation = f"Error executing tool {action}: {e}"
            else:
                observation = f"Error: Tool '{action}' not found. Available: {list(tools.AVAILABLE_TOOLS.keys())}"

            obs_preview = observation[:self.obs_window]
            print(f"[AGENT OBSERVATION] {obs_preview[:200]}...")
            history.append(f"Observation: {obs_preview}")
            yield {"type": "observation", "content": obs_preview + ("..." if len(observation) > self.obs_window else "")}

        print("[AGENT ERROR] Max steps reached.")
        yield {"type": "error", "content": "The agent was unable to find a definitive answer within the allotted steps."}
