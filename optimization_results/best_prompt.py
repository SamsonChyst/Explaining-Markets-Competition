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
