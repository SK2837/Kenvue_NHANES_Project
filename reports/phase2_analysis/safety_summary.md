**Elevated ALT in U.S. Adults: Key Risk Factors and Prevalence Findings (NHANES 2017–2018)**

## 1. What We Studied
We analyzed data from 3,543 U.S. adults in the National Health and Nutrition Examination Survey (NHANES 2017–2018) to identify who is most at risk for elevated alanine transaminase (ALT), a liver enzyme that signals liver stress or injury when elevated above 40 U/L.

## 2. Who Is Most Affected
About 5% of U.S. adults have elevated ALT levels. Risk varies considerably by group:
- Adults aged 18–40 are at highest risk (8.7% prevalence vs. 2.7% in adults 60 and older)
- Mexican Americans (8.9%) and Non-Hispanic Asians (7.0%) show the highest prevalence by race/ethnicity
- Adults with obesity show elevated rates (8.3%) compared to those at normal weight (2.0%)
- Women have slightly higher raw prevalence (6.5% vs. 3.9% in men), but in adjusted models, male sex shows higher odds

## 3. Key Risk Factors
Our updated statistical analysis tested 13 candidate variables. Three factors emerged as consistently significant independent predictors across model specifications:

**Confirmed significant predictors:**
- **Larger waist circumference** — each centimeter increases adjusted risk by ~3%. Belly fat (visceral fat) is more harmful to the liver than overall body weight, because it is processed directly through the liver.
- **Higher triglycerides** — people with triglycerides of 200 mg/dL face ~27% higher adjusted odds than those at 100 mg/dL. Elevated blood fats are a key signal of metabolic stress on the liver.
- **Younger age** — risk decreases with age (likely survivor bias: those at high risk develop disease earlier).

**Not significant after full adjustment:**
- Blood levels of lead, cadmium, and mercury — no significant association found across any model. This is an important null finding for occupational and environmental safety assessment.
- Alcohol intake and college education — tested as new variables; neither reached significance after adjusting for metabolic factors.

## 4. Individual Risk Screening
We built a machine learning tool (XGBoost) to estimate individual risk using a broader set of clinical markers. Top predictors of elevated ALT in this individual-level model include:
- Liver enzymes AST and GGT (most powerful — they co-move with ALT)
- Age
- **Triglyceride levels** (consistent with the epidemiologic model)
- **Waist circumference** (consistent with the epidemiologic model)
- Blood glucose, albumin

The agreement between the ML model and the epidemiologic model on waist circumference and triglycerides strengthens confidence in these findings.

## 5. The Metabolic Pathway
Our findings point to a clear biological story: **visceral fat → elevated triglycerides → liver fat accumulation → elevated ALT**. This is the non-alcoholic fatty liver disease (NAFLD) pathway, and it is now confirmed as the dominant driver in this nationally representative U.S. population sample.

## 6. Implications for Safety Monitoring
- Waist circumference and triglycerides are inexpensive, routinely measured clinical markers that can serve as early signals for liver stress
- Safety monitoring programs should prioritize metabolic health screening alongside liver enzyme testing
- The null finding for heavy metals suggests that environmental toxicant exposure — at current U.S. population exposure levels — is not a primary driver of ALT elevation in the general population
- This tool is a research estimate, not a clinical diagnostic. Consult a physician for individual health decisions.
