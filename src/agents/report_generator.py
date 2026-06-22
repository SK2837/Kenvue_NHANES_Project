"""
Report Generator — produces structured health risk reports on demand.

Given a person profile or topic, generates a full markdown report covering:
  demographics, health conditions, risk assessment, population context, recommendations.
"""

from __future__ import annotations

import json
import os

import anthropic

from .tools import ANALYSIS_TOOL_NAMES, execute_tool, get_tool_schemas

_SYSTEM_PROMPT = """\
You are a clinical data analyst generating structured health risk reports based on the
NHANES 2017–2018 liver injury study (N=3,543 U.S. adults, survey-weighted).

When asked to generate a report, ALWAYS:
1. Call the relevant tools first to get real data (compute_individual_risk, read_or_table,
   read_prevalence_table, read_shap_rankings) — never use made-up numbers.
2. Then produce a thorough, well-formatted markdown report using the structure below.

═══ REQUIRED REPORT STRUCTURE ═══

# [Report Title]
*Generated from NHANES 2017–2018 | Survey-weighted analysis*

---

## Executive Summary
2–3 sentence overview of the subject/topic and the single most important finding.

---

## 1. Subject Demographics
- If a person profile was provided: describe their demographic profile in the context of
  the U.S. adult population (what percentile of BMI, age group, how common is diabetes
  in this age/sex group, etc.)
- If a group was queried: describe the group size, U.S. representation, and key traits.

## 2. Health Conditions Overview
- Explain the relevant health conditions (diabetes, obesity, etc.) in this context.
- Include weighted prevalence from the NHANES data for this group.
- Use actual numbers from the tools.

## 3. Health Risk Assessment
- Individual risk estimate % (use compute_individual_risk if person profile given)
- Compare to U.S. adult average (5.2%)
- Relative odds vs population average
- Top 3 risk drivers explained in plain language

## 4. Population Context
- How does this person/group compare across the broader NHANES population?
- Subgroup comparisons: sex, age, BMI, diabetes — use real prevalence numbers.
- Convergence between ML model (SHAP) and population model (OR) findings.

## 5. Key Findings & Implications
- 3–5 bullet points of actionable insights
- What modifiable risk factors are present?
- What does the science say about reducing these risks?

## 6. Statistical Notes & Limitations
- Methodology: survey-weighted logistic regression, Taylor-series CIs
- Cross-sectional design: associations only, not causal
- Disclaimer: statistical estimate, not a clinical test
- Model confidence and applicable population

---
*Data source: NHANES 2017–2018 | Analysis: Survey-weighted logistic regression + XGBoost*

═══ TONE & STYLE ═══
- Professional but accessible — readable by a non-statistician
- Use **bold** for key numbers and findings
- Use tables where helpful for comparisons
- Never use jargon without explanation
- Always end with the disclaimer that this is a research estimate, not medical advice
"""


class ReportGenerator:
    MODEL     = "claude-sonnet-4-6"
    MAX_TURNS = 6

    def __init__(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY environment variable not set.")
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, request: str) -> str:
        """Generate a structured report for a given profile or topic.

        request: free-text description, e.g.
            "Generate a report for a 52-year-old diabetic male with BMI 34"
            "Generate a population report on obesity and liver risk"
        Returns full markdown report as a string.
        """
        messages = [{"role": "user", "content": f"Please generate a full health risk report for: {request}"}]
        schemas  = get_tool_schemas(ANALYSIS_TOOL_NAMES)

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
        ) or "Report generation failed. Please try again."
