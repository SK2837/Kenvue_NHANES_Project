"""
Orchestrator — routes user messages to QAAgent or AnalysisAgent.

Uses a single fast routing call (no tools, low token count) to classify the
question, then delegates to the appropriate specialist agent.
"""

from __future__ import annotations

import json
import os
import re

import anthropic

from .analysis_agent import AnalysisAgent
from .qa_agent import QAAgent

_ROUTING_PROMPT = """\
You route user questions about a NHANES liver injury study to the right agent.
Return ONLY valid JSON on a single line — no other text, no markdown.

Format: {"agent": "qa" | "analysis" | "direct", "reason": "one sentence"}

Routing rules:
→ "qa": personalized risk questions (user describes a person: BMI/diabetes/age/sex),
        stats lookups (what is the OR for BMI, top SHAP features, AUC-ROC, prevalence
        for groups already in the table like males/females/age groups), report retrieval.

→ "analysis": custom subgroup queries (NOT already in the prevalence table, e.g. Hispanic
              males over 40), run/re-run pipeline scripts, subgroup logistic regression
              ("run the analysis for diabetic women"), in-depth "why" or "what causes"
              mechanistic explanations, comparing multiple subgroup results.

→ "direct": greetings ("hi", "hello"), meta-questions about the system itself
            ("what can you do?", "which agents exist?"), "thank you", "bye".

When uncertain between qa and analysis, choose "qa" (cheaper and faster).
"""


class Orchestrator:
    MODEL = "claude-haiku-4-5-20251001"

    def __init__(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY environment variable not set.")
        self.client    = anthropic.Anthropic(api_key=api_key)
        self._qa       = QAAgent()
        self._analysis = AnalysisAgent()

    def chat(
        self,
        user_message: str,
        history: list[dict] | None = None,
    ) -> tuple[str, str]:
        """Route and respond.

        Returns (response_text, agent_name) where agent_name is
        'Q&A Agent', 'Analysis Agent', or 'Orchestrator'.
        """
        history = history or []

        # ── 1. Route ──────────────────────────────────────────────────────────
        agent_key = self._route(user_message, history)

        # ── 2. Dispatch ───────────────────────────────────────────────────────
        if agent_key == "direct":
            response = self._direct_reply(user_message)
            return response, "Orchestrator"

        if agent_key == "analysis":
            response = self._analysis.run(user_message, history)
            return response, "Analysis Agent"

        # Default: qa
        response = self._qa.run(user_message, history)
        return response, "Q&A Agent"

    def _route(self, user_message: str, history: list[dict]) -> str:
        """Return 'qa', 'analysis', or 'direct'."""
        # Build compact history context (last 2 turns max) for routing
        context_turns = history[-4:] if len(history) > 4 else history
        routing_messages = context_turns + [{"role": "user", "content": user_message}]

        try:
            response = self.client.messages.create(
                model=self.MODEL,
                system=_ROUTING_PROMPT,
                messages=routing_messages,
                max_tokens=100,
            )
            text = response.content[0].text.strip() if response.content else ""
            # Extract JSON even if there's surrounding text
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return data.get("agent", "qa")
        except Exception:
            pass

        return "qa"  # fallback

    def _direct_reply(self, user_message: str) -> str:
        """Handle greetings and meta-questions directly without tools."""
        response = self.client.messages.create(
            model=self.MODEL,
            system=(
                "You are a helpful assistant for a NHANES liver injury analysis project. "
                "Answer greetings and meta-questions briefly. "
                "For meta-questions about capabilities: explain that you can (1) estimate "
                "individual ALT elevation risk given a person's profile, (2) look up study "
                "statistics (ORs, SHAP rankings, prevalence by group, ML metrics), "
                "(3) retrieve stakeholder reports, and (4) run custom subgroup analyses. "
                "Keep responses under 100 words."
            ),
            messages=[{"role": "user", "content": user_message}],
            max_tokens=200,
        )
        return response.content[0].text if response.content else "Hello! How can I help?"
