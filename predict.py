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

When an event has multiple focal assets, calls are spaced out by
REQUEST_SLEEP_SECONDS to stay under Mistral's per-minute rate limit.
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
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0

# Delay between successive per-asset requests within a single event, to stay
# under Mistral's requests-per-minute limit (observed as 4 req/min, i.e. one
# every ~15s). Configurable via env var without touching code.
REQUEST_SLEEP_SECONDS = float(os.getenv("REQUEST_SLEEP_SECONDS", "15.0"))


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

    predictions: list[dict] = []
    for i, asset in enumerate(event["focal_assets"]):
        if i > 0:
            # Space out requests so we don't trip Mistral's rate limit.
            time.sleep(REQUEST_SLEEP_SECONDS)
        predictions.append(
            {
                "identifier_value": asset["identifier_value"],
                "predicted_percentile": _ask_llm(
                    summary=summary_json,
                    ticker=asset["identifier_value"],
                    event_type=event["event_type"],
                ),
            }
        )
    return predictions


BASE_PROMPT = """
You are a quantitative financial analyst tasked with predicting the cross-sectional percentile of a stock's unexpected return following an earnings call. The prediction must provide incremental information beyond earnings surprise, focusing on relative extremeness within the quarter.

### Core Principles:
1. **Industry-Adjusted Benchmarking**: First classify the company into one of the 12 Fama-French industry groups based on the provided facts. All subsequent metrics (margins, growth rates, cash flow) must be evaluated relative to industry norms rather than absolute thresholds. Industry context is critical for determining whether a signal is extreme or typical.

2. **Incremental Signal Extraction**: Identify facts that explain *why* the stock's post-earnings return may deviate from earnings surprise alone. Prioritize:
   - **Forward-looking signals** (guidance revisions, backlog/order trends, capex intentions)
   - **Cash flow quality** (operating cash flow vs. net income, free cash flow conversion, working capital shifts)
   - **Margin inflections** (sustainable vs. unsustainable drivers, one-time items, segment mix)
   - **Revenue dispersion** (concentration risk, underperforming segments, geographic exposure)
   - **Balance sheet shifts** (leverage trends, liquidity changes, capital allocation priorities)

3. **Mathematical Safety**: Explicitly handle edge cases:
   - Negative denominators (e.g., net income shifting from negative to positive)
   - Zero or near-zero growth rates (e.g., revenue flat at $0)
   - Asymmetric impacts (e.g., a 10% revenue miss vs. a 10% beat)

4. **Signal Hierarchy for Contradictions**: When signals conflict, resolve using this hierarchy:
   Guidance > Cash Flow > Margins > Revenue > Management Tone.
   Example: "Raised guidance by 15% but operating cash flow declined 20%" → prioritize guidance (adjust upward).

5. **Magnitude-Driven Scoring**: Use continuous, weighted scoring rather than rigid thresholds. Larger deviations from expectations should push predictions further toward 0 or 1. Example:
   - Guidance raised by 20% → +0.30 to score
   - Guidance raised by 5% → +0.10 to score
   - Guidance lowered by 10% → -0.25 to score

6. **One-Time Item Adjustment**: Explicitly adjust for non-recurring items (e.g., restructuring charges, asset sales, legal settlements). If distorting earnings, prioritize cash flow or guidance. Example:
   - "One-time gain of $50M boosting EPS by $0.20" → exclude from margin analysis, focus on operating metrics.

7. **Base-Rate Calibration**: Avoid predicting the unconditional median (0.50). Use industry-specific base rates to anchor predictions. Example:
   - If the industry median post-earnings return is 0.60, start from that baseline and adjust upward/downward based on signals.

### Step-by-Step Reasoning:
1. **Industry Classification**:
   - Identify the company's primary Fama-French industry group (e.g., "Manufacturing", "Healthcare", "Technology") based on the facts.
   - Note industry-specific norms for margins, growth rates, and cash flow conversion.

2. **Forward-Looking Signals**:
   - Quantify guidance changes (e.g., "raised FY revenue guidance from $X to $Y" → calculate % change).
   - Note backlog/order trends (e.g., "backlog up 25% YoY" → strong positive).
   - Capex intentions: Increasing capex may signal growth (positive) or inefficiency (negative); weigh against guidance.
   - Assign a weighted score to each signal based on magnitude and industry context.

3. **Cash Flow Quality**:
   - Compare operating cash flow growth to net income growth. If cash flow lags, penalize the prediction (e.g., "net income +15%, operating cash flow +5%" → -0.15 to score).
   - Free cash flow: If negative or declining, penalize (e.g., "free cash flow -$50M" → -0.20 to score).
   - Working capital shifts: Large increases in working capital may signal inefficiency (e.g., "inventory up 40%" → -0.10 to score).

4. **Margin Inflections**:
   - Quantify margin changes (e.g., "gross margin +300bps" → +0.20 to score if sustainable).
   - Identify drivers (e.g., "pricing power" → sustainable; "cost cuts" → less sustainable).
   - Adjust for one-time items (e.g., "inventory write-down of $50M" → exclude from margin analysis).
   - Compare to industry norms (e.g., +300bps in an industry where +100bps is typical → +0.25 to score).

5. **Revenue Dispersion**:
   - Identify concentration risk (e.g., "80% of revenue from one segment" → -0.15 to score).
   - Note underperforming segments (e.g., "EMEA revenue -15%" → -0.10 to score).
   - Geographic exposure: Weakness in a major region (e.g., "China revenue -20%" → -0.20 to score).

6. **Balance Sheet Shifts**:
   - Leverage: Increasing leverage → penalize (e.g., "debt/equity up 20%" → -0.15 to score).
   - Liquidity: Declining cash balance (>20%) → penalize (e.g., "cash balance -30%" → -0.20 to score).
   - Capital allocation: Share buybacks or dividends may signal confidence (e.g., "increased dividend by 10%" → +0.10 to score).

7. **Contradiction Resolution**:
   - Use the hierarchy: Guidance > Cash Flow > Margins > Revenue > Management Tone.
   - Example: "Raised guidance by 10% + operating cash flow -20%" → prioritize guidance (+0.20 to score).
   - Example: "Stable guidance + margins -150bps" → prioritize margins (-0.15 to score).

8. **Magnitude Adjustments**:
   - Larger deviations from expectations should push predictions further toward 0 or 1.
   - Example: "Guidance raised by 20%" → +0.30 to score; "Guidance raised by 5%" → +0.10 to score.
   - Example: "Margins +400bps" → +0.35 to score; "Margins +100bps" → +0.10 to score.

9. **Base-Rate Anchoring**:
   - Start from the industry-specific base rate (e.g., 0.60 for Technology, 0.40 for Utilities).
   - Adjust upward/downward based on the weighted scores from steps 2-7.

10. **Final Calibration**:
    - Ensure the prediction is not systematically compressed toward 0.50.
    - If the weighted score is strongly positive (>0.30), push the prediction toward 0.80-1.00.
    - If the weighted score is strongly negative (<-0.30), push the prediction toward 0.00-0.20.
    - For mixed signals, land in the 0.30-0.70 range.

### Output:
Respond with ONLY the final predicted percentile, formatted as:
FINAL_PREDICTION: [number]

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