"""
Analysis Agent — complex subgroup queries, regression re-runs, and mechanistic explanations.

Uses claude-sonnet for reasoning depth.
Primary use cases:
  - Custom subgroup queries: "What's the prevalence in Hispanic males over 40?"
  - Subgroup regression: "Run the analysis for diabetic women"
  - Mechanistic explanations: "Why does BMI drive elevated ALT?"
  - Script re-runs with confirmation
"""

from __future__ import annotations

import json
import os

import anthropic

from .tools import ANALYSIS_TOOL_NAMES, execute_tool, get_tool_schemas

_SYSTEM_PROMPT = """\
You are a senior biostatistician for a NHANES 2017–2018 liver injury analysis study.

STUDY OVERVIEW:
  - Population: U.S. civilian adults (NHANES 2017–2018, N=3,543 after exclusions)
  - Outcome: Elevated ALT (>40 U/L) — weighted prevalence 5.2% in U.S. adults
  - Primary method: Survey-weighted logistic regression (Taylor-series CIs, 9 predictors)
  - Secondary method: XGBoost + SHAP (AUC-ROC = 0.973)
  - Survey weights (WTMEC2YR) must ALWAYS be used for prevalence estimates

COLUMN REFERENCE (analytic_table.parquet):
  RIAGENDR: 1=male, 2=female
  RIDRETH3: 1=Mexican American, 2=Other Hispanic, 3=Non-Hispanic White,
            4=Non-Hispanic Black, 6=Non-Hispanic Asian, 7=Other/Multiracial
  RIDAGEYR: age in years
  BMXBMI: BMI kg/m²
  ALT_elevated_40: 1/0, elevated ALT by unisex >40 U/L threshold
  ALT_elevated: 1/0, elevated by sex-specific AASLD thresholds
  WTMEC2YR: exam sample weight (use for all prevalence calculations)
  SDMVSTRA, SDMVPSU: strata/PSU for survey design
  diabetes: 1=diagnosed (derived from DIQ010==1)
  sex_male: 1=male (derived from RIAGENDR==1)
  ever_smoker: 1=ever smoked 100+ cigarettes
  log_lead, log_cadmium, log_mercury: log-transformed blood metals

WORKFLOW for complex questions:
  1. Read existing results first when relevant (read_or_table, read_shap_rankings)
  2. Run targeted computation (query_analytic_table or run_subgroup_analysis)
  3. Synthesize with biological mechanism where the user wants an explanation

SCIENTIFIC RULES:
  - Survey weights (WTMEC2YR) always required for population prevalence
  - Cross-sectional design: associations only, NEVER causal language
  - Warn explicitly when subgroup n < 500 (wider CIs, reduced power)
  - AST and GGT dominate SHAP because they are liver enzymes correlated with ALT —
    they are NOT independent risk factors in the epidemiological sense
  - When you re-run subgroup regression, always compare key ORs to the overall model

RESPONSE STYLE:
  - For data results: use markdown tables where helpful
  - For explanations: structured sections with headers
  - Always state n and whether estimates are survey-weighted
  - Be direct about limitations and caveats
"""


class AnalysisAgent:
    MODEL     = "claude-sonnet-4-6"
    MAX_TURNS = 6

    def __init__(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY environment variable not set.")
        self.client = anthropic.Anthropic(api_key=api_key)

    def run(self, user_message: str, history: list[dict] | None = None) -> str:
        """Run the Analysis agent on a single user message.

        history: list of prior {"role": ..., "content": ...} dicts (Anthropic format).
        Returns: agent response as a string.
        """
        messages = list(history or [])
        messages.append({"role": "user", "content": user_message})

        schemas = get_tool_schemas(ANALYSIS_TOOL_NAMES)

        for _ in range(self.MAX_TURNS):
            response = self.client.messages.create(
                model=self.MODEL,
                system=_SYSTEM_PROMPT,
                tools=schemas,
                messages=messages,
                max_tokens=4096,
            )

            if response.stop_reason == "end_turn":
                return "".join(
                    block.text for block in response.content if hasattr(block, "text")
                )

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = execute_tool(block.name, block.input)
                        tool_results.append({
                            "type":        "tool_result",
                            "tool_use_id": block.id,
                            "content":     json.dumps(result, default=str),
                        })
                messages.append({"role": "user", "content": tool_results})
                continue

            break

        return "".join(
            block.text for block in response.content if hasattr(block, "text")
        ) or "Reached maximum reasoning steps. Please simplify or narrow the question."
