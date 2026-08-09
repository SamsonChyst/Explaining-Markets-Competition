"""
'predict(event)` is called once per competition event, after the webhook has
already been verified for you. Return one prediction per focal asset. Everything
else in this repo (webhook verification, dedupe, submission) is plumbing.

This implementation asks a Mistral model for a calibrated percentile via one
chat-completion call per asset. `MISTRAL_KEY` is required — the module raises
immediately at import time if it's missing, so a missing key is caught at
deploy time rather than mid-run. A single failed call to the Mistral API
(network/API error, or a reply that can't be parsed into a number) is retried
a few times before falling back to a 0.5 placeholder for that one asset, so
one bad event doesn't stall the whole prediction run.
"""

from __future__ import annotations
import json
import os
import re
import time
import httpx
from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()
MISTRAL_KEY = os.getenv("MISTRAL_KEY")

if not MISTRAL_KEY:
    raise EnvironmentError(
        "MISTRAL_KEY is not set in the environment."
    )

client = Mistral(api_key=MISTRAL_KEY)
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-latest")

# One _ask_llm call retries this many times (including the first attempt)
# before giving up and returning the 0.5 placeholder.
MAX_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = 2.0


def predict(event: dict) -> list[dict]:
    """Return predictions for one Explaining Markets event.

    `event` is the verified webhook payload. Useful fields:
      event["event_type"]          e.g. "EARNINGS_RELEASE"
      event["focal_assets"]        list of {"identifier_type", "identifier_value"}
      event["information_url"]     short-lived signed URL with the event summary JSON
      event["prediction_deadline"] ISO timestamp; submit before this fires

    Required return: a list of dicts, one per focal asset:
      [{"identifier_value": "AAPL", "predicted_percentile": 0.71}, ...]

    `predicted_percentile` is a float in [0, 1] — your prediction of where the
    asset's next-day *unexpected* return will fall in its historical distribution
    (0 = worst, 0.50 = median, 1 = best).
    """
    summary = httpx.get(event["information_url"], timeout=10.0)
    summary.raise_for_status()
    summary_json = summary.json()

    print(summary_json)
    return [
        {
            "identifier_value": asset["identifier_value"],
            "predicted_percentile": _ask_llm(
                summary=summary_json,
                ticker=asset["identifier_value"],
                event_type=event["event_type"],
            ),
        }
        for asset in event["focal_assets"]
    ]


BASE_PROMPT = """
You are a quantitative financial analyst tasked with predicting the cross-sectional percentile of a stock's unexpected return following an earnings call. The prediction must provide incremental information beyond earnings surprise, focusing on relative extremeness within the quarter. Output only a number between 0 and 1.

### Core Principles:
1. **Incremental Signal Extraction**: Identify facts that explain *why* the stock's post-earnings return may deviate from earnings surprise alone. Prioritize:
   - **Forward-looking signals** (guidance changes, outlook revisions, backlog trends)
   - **Cash flow dynamics** (operating cash flow vs. net income, capex changes, working capital shifts)
   - **Margin sustainability** (pricing power vs. cost cuts, mix shifts, one-time items)
   - **Segment dispersion** (concentration risk, underperforming segments)
   - **Balance sheet health** (leverage trends, liquidity changes)

2. **Cross-Sectional Ranking**: Predict how extreme the stock's performance will be *relative to peers in the same quarter*. Key adjustments:
   - **Magnitude matters**: Larger deviations (e.g., "gross margin +400bps" vs "+100bps") should push predictions further toward 0 or 1.
   - **Contradiction resolution**: When signals conflict, use this hierarchy: guidance > cash flow > margins > revenue > management tone.
   - **One-time items**: Explicitly adjust for non-recurring items (e.g., "restructuring charge of $X million"). If distorting earnings, prioritize cash flow.

3. **Calibration Rules**:
   - **Top 25% (0.75-1.00)**: Requires *all* of:
     - Raised guidance (>5% for revenue, >10% for EPS) *or* strong backlog growth (>15%)
     - Expanding margins (>200bps) with sustainable drivers (pricing power, volume)
     - Positive operating cash flow growth > net income growth
     - No major contradictions (e.g., weak segment performance)
   - **Bottom 25% (0.00-0.25)**: Requires *any* of:
     - Lowered guidance (>5% for revenue, >10% for EPS) *or* backlog decline (>10%)
     - Contracting margins (>100bps) with unsustainable drivers (cost cuts, mix shifts)
     - Operating cash flow growth < net income growth *or* negative free cash flow
     - Liquidity concerns (cash balance decline >20%, leverage increase)
   - **Middle 50% (0.25-0.75)**: All other cases, with adjustments for:
     - Mixed signals (e.g., stable guidance + weak cash flow → 0.4-0.6)
     - One-time items (e.g., asset sale gain distorting earnings → adjust toward 0.5)
     - Segment dispersion (e.g., strong core segment + weak peripheral → 0.3-0.7)

### Step-by-Step Reasoning:
1. **Forward-Looking Signals**:
   - Quantify guidance changes (e.g., "raised FY revenue guidance from $X to $Y" → calculate % change).
   - Note backlog/order trends (e.g., "backlog up 20% YoY" → strong positive).
   - Ignore generic management commentary unless tied to specific metrics.

2. **Cash Flow Analysis**:
   - Compare operating cash flow growth to net income growth. If cash flow lags, penalize the prediction (e.g., "net income +15%, operating cash flow +5%" → adjust downward).
   - Note capex changes (e.g., "capex increased 30%" → may signal growth or inefficiency; weigh against guidance).
   - Free cash flow: If negative or declining, penalize the prediction.

3. **Margin Sustainability**:
   - Quantify margin changes (e.g., "gross margin +300bps" → strong positive).
   - Identify drivers (e.g., "pricing power" → sustainable; "cost cuts" → less sustainable).
   - Adjust for one-time items (e.g., "inventory write-down of $50M" → exclude from margin analysis).

4. **Segment Performance**:
   - Identify concentration risk (e.g., "80% of revenue from one segment" → adjust downward).
   - Note underperforming segments (e.g., "EMEA revenue -15%" → adjust downward).

5. **Balance Sheet Health**:
   - Leverage: Increasing leverage → adjust downward.
   - Liquidity: Declining cash balance (>20%) → adjust downward.

6. **Contradiction Resolution**:
   - Use the hierarchy: guidance > cash flow > margins > revenue.
   - Example: "Raised guidance by 10% + weak cash flow" → prioritize guidance (adjust toward 0.8-0.9).
   - Example: "Stable guidance + operating cash flow -20%" → prioritize cash flow (adjust toward 0.3-0.4).

7. **Magnitude Adjustments**:
   - Larger deviations from expectations should push predictions further toward 0 or 1.
   - Example: "Guidance raised by 20%" → 0.9; "Guidance raised by 5%" → 0.7.

### Prediction Rules:
- **Top 25% (0.75-1.00)**:
  - Raised guidance (>10% EPS or >5% revenue) *or* backlog growth >15%.
  - Expanding margins (>200bps) with sustainable drivers.
  - Operating cash flow growth > net income growth.
  - No major contradictions.
- **Bottom 25% (0.00-0.25)**:
  - Lowered guidance (>10% EPS or >5% revenue) *or* backlog decline >10%.
  - Contracting margins (>100bps) with unsustainable drivers.
  - Operating cash flow growth < net income growth *or* negative free cash flow.
  - Liquidity concerns (cash balance decline >20%).
- **Middle 50% (0.25-0.75)**:
  - Mixed signals (e.g., stable guidance + weak cash flow → 0.4-0.6).
  - One-time items (e.g., asset sale gain → adjust toward 0.5).
  - Segment dispersion (e.g., strong core + weak peripheral → 0.3-0.7).

### Output:
Respond with ONLY a number between 0 and 1 representing the predicted percentile.

Key facts from the earnings call:
{facts}
"""


def _ask_llm(*, summary: dict, ticker: str, event_type: str) -> float:
    """Ask Mistral for a calibrated percentile for one asset.

    Retries transient failures (API/network errors, or a reply that can't be
    parsed into a [0, 1] float) up to MAX_ATTEMPTS times with a short backoff.
    If every attempt fails, returns 0.5 for that asset so one bad event never
    stalls the whole prediction run.
    """
    summary_text = summary.get("summary") if isinstance(summary, dict) else None
    if not summary_text:
        summary_text = json.dumps(summary)
    summary_text = summary_text[:8000]

    facts = f"Event type: {event_type}\nTicker: {ticker}\n\n{summary_text}"
    prompt = BASE_PROMPT.format(facts=facts)

    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.chat.complete(
                model=MISTRAL_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            return _parse_percentile(response.choices[0].message.content)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            last_error = exc
            if attempt < MAX_ATTEMPTS:
                print(
                    f"[WARN] Mistral call failed for {ticker} "
                    f"(attempt {attempt}/{MAX_ATTEMPTS}): {exc!r}"
                )
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    print(
        f"[WARN] Mistral call failed for {ticker} after {MAX_ATTEMPTS} attempts "
        f"({last_error!r}) — submitting 0.5 placeholder."
    )
    return 0.5


def _parse_percentile(content: str | None) -> float:
    """Parse the model's raw reply into a float clamped to [0, 1].

    Raises ValueError if no number can be found, so `_ask_llm`'s retry loop
    treats an unparseable reply the same as a failed API call.
    """
    if not content:
        raise ValueError("empty response from Mistral")

    match = re.search(r"-?\d*\.?\d+", content.strip())
    if not match:
        raise ValueError(f"could not find a number in Mistral response: {content!r}")

    return max(0.0, min(1.0, float(match.group())))