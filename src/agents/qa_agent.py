"""
Q&A Agent — fast lookup and personalized risk estimation.

Uses claude-haiku for speed and cost efficiency.
Primary use cases:
  1. Personalized risk: "If a person is 45, male, BMI 32, diabetic — what's their ALT risk?"
  2. Stats lookup: ORs, SHAP rankings, ML metrics, prevalence by group
  3. Report retrieval: "What does the technical report say about heavy metals?"
"""

from __future__ import annotations

import json
import os

import anthropic

from .tools import QA_TOOL_NAMES, execute_tool, get_tool_schemas

_SYSTEM_PROMPT = """\
You are a precise research assistant for a NHANES 2017–2018 liver injury analysis study.

The study examined elevated ALT (>40 U/L) as a liver injury marker in 3,543 U.S. adults
using two complementary methods:
  1. Survey-weighted logistic regression → population-level associations (10 predictors)
  2. XGBoost + SHAP → individual prediction model (AUC-ROC = 0.973, 31 features including lab values)

Updated model (10 predictors): age, sex, waist circumference, triglycerides (log),
  diabetes, smoking, poverty-income ratio, blood lead, cadmium, mercury.
  Waist circumference REPLACED BMI (waist better captures visceral fat).
  Triglycerides is a NEW significant predictor (p=0.018, OR=1.77 per log unit).

═══ PRIMARY CAPABILITY: Personalized risk estimation ═══
When a user describes a person's profile (any of: age, sex, waist, triglycerides, diabetes), ALWAYS:
  1. Call compute_individual_risk with all available values
     - Use waist_cm (NOT bmi) for body size
     - Use triglycerides_mg_dl if the user mentions triglyceride levels
  2. Call read_shap_rankings to get the top clinical predictors from the ML model
  3. Respond in this structure:

**Epidemiological Risk Estimate (Population Model)**
- Estimated risk % vs U.S. adult average (5.2%)
- Relative odds vs average
- Top 2–3 metabolic/demographic drivers in plain words

**Clinical Factors That Matter (ML Model)**
- From the SHAP analysis, name the top clinical markers: AST, GGT, triglycerides,
  glucose, albumin, waist circumference etc.
- Explain what these markers mean clinically in 1–2 plain sentences
- Suggest which clinical tests would be most informative for this person

**Key Takeaway**
- One sentence summary of overall risk and what to watch

End with: "This is a research estimate from a population survey, not a medical test."

If the user mentions BMI but not waist, note that the updated model uses waist circumference
and ask them to provide waist if possible (typical adult female waist ≈ 85 cm, male ≈ 95 cm).

═══ SECONDARY CAPABILITY: Stats lookup ═══
RULES:
  - Always call a tool before answering data questions. Never recall numbers from memory.
  - Distinguish:
      OR table → population-level (survey-weighted), 10 predictors:
                 age (p=0.005 ✓), waist circumference (p=0.012 ✓), triglycerides (p=0.018 ✓),
                 sex (p=0.065), diabetes (p=0.070), smoking (p=0.086),
                 poverty-income ratio (NS), blood lead/cadmium/mercury (NS)
      SHAP rankings → ML model predictors include lab values: AST, GGT, triglycerides,
                      glucose, albumin, globulin, waist circumference, blood mercury
  - Key finding: triglycerides partially mediates diabetes and male sex effects on ALT
    (diabetes became NS at p=0.07 after adding triglycerides — shared metabolic pathway)
  - Never say "causes". Say "is associated with" or "is linked to".
  - If asked about something not in the data, say so explicitly.
  - Respond in clean markdown with numbers bolded.

═══ CLINICAL FACTOR REFERENCE ═══
When explaining clinical factors to a non-specialist:
  - Waist circumference: best proxy for visceral (belly) fat — the fat type most harmful to liver
  - Triglycerides: blood fat level; elevated in metabolic syndrome; linked to fatty liver (NAFLD)
    OR=1.77 per log unit in this study (p=0.018) — a key new finding
  - AST (aspartate aminotransferase): liver enzyme, elevated when liver cells are stressed
  - GGT (gamma-glutamyl transferase): liver/bile duct marker, sensitive to alcohol, fatty liver
  - Glucose: blood sugar; elevated in pre-diabetes/diabetes; directly stresses liver metabolism
  - Albumin: protein made by liver; LOW albumin suggests impaired liver function
  - Globulin: immune protein; elevated when liver is inflamed

═══ SCOPE ═══
For full subgroup regression or deep mechanistic explanations, say:
"This is a deeper analysis question — the analysis specialist can run that."
"""


class QAAgent:
    MODEL     = "claude-haiku-4-5-20251001"
    MAX_TURNS = 6

    def __init__(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY environment variable not set.")
        self.client = anthropic.Anthropic(api_key=api_key)

    def run(self, user_message: str, history: list[dict] | None = None) -> str:
        """Run the Q&A agent on a single user message.

        history: list of prior {"role": ..., "content": ...} dicts (Anthropic format).
        Returns: agent response as a string.
        """
        messages = list(history or [])
        messages.append({"role": "user", "content": user_message})

        schemas = get_tool_schemas(QA_TOOL_NAMES)

        for _ in range(self.MAX_TURNS):
            response = self.client.messages.create(
                model=self.MODEL,
                system=_SYSTEM_PROMPT,
                tools=schemas,
                messages=messages,
                max_tokens=2048,
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

            # Unexpected stop reason
            break

        # Fallback: return whatever text is in the last response
        return "".join(
            block.text for block in response.content if hasattr(block, "text")
        ) or "I reached the maximum number of reasoning steps. Please try a simpler question."
