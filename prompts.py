
BASE_PROMPT = """Predict the unexpected stock return following an earnings call.

You are given key facts from a company's earnings call transcript.
Predict the stock's unexpected return as a class and percentile.

Base rates — calibrate your predictions to these proportions:
  - ~25% of stocks go UP (price increases 5%+ after the call)
  - ~50% of stocks are NEUTRAL (price moves less than 5%)
  - ~25% of stocks go DOWN (price decreases 5%+ after the call)

Consistency constraints between class and percentile:
  - "down"    → percentile in [0.00, 0.25]
  - "neutral" → percentile in [0.25, 0.75]
  - "up"      → percentile in [0.75, 1.00]

Your rationale must reference substantive evidence directly
(e.g., "Revenue grew 18% year-over-year…"). Never reference fact
numbers (e.g., never say "fact 3 shows…" or "according to fact 7").

Key facts from the earnings call:
{facts}

Respond with ONLY a number between 0 and 1 for the predicted percentile.
"""