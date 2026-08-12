import asyncio
import json
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Optional
import pandas as pd
import statsmodels.api as sm
from dotenv import load_dotenv
from mistralai.client import Mistral
from prompts import BASE_PROMPT


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

MISTRAL_KEY = os.getenv("MISTRAL_KEY")

if not MISTRAL_KEY:
    raise EnvironmentError(
        "MISTRAL_KEY is not set in the environment."
    )

client = Mistral(api_key=MISTRAL_KEY)


# ------------------------------------------------------------
# Files
# ------------------------------------------------------------

DATA_PATH = Path("dataset_cleaned.csv")

OUTPUT_DIR = Path("optimization_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_PROMPT_PATH = OUTPUT_DIR / "best_prompt.py"
STATE_PATH = OUTPUT_DIR / "optimization_state.json"
HISTORY_PATH = OUTPUT_DIR / "optimization_history.csv"


# ------------------------------------------------------------
# Models
# ------------------------------------------------------------

MODEL = "mistral-large-latest"

# Same Large model is used both for:
# 1. predictions
# 2. prompt evolution
#
# If you later have another model available, it is better to
# use a cheaper/faster model for the optimizer.


# ------------------------------------------------------------
# Evolution — SEARCH (train) sample
# ------------------------------------------------------------

GENERATIONS = 15
CANDIDATES_PER_GENERATION = 2

# The search/train sample is stratified by DECILE of
# target_percentile_quarter (not by quarter anymore). From every
# decile we draw a random number of events (per_decile_min.max),
# preferring "inconsistent" events -- ones where
# surprise_percentile_quarter poorly explains
# target_percentile_quarter. Those are exactly the events that
# carry incremental signal beyond earnings surprise, which is
# what the prompt is being optimized to extract.
#
# Example:
#   10 deciles x ~16.5 average -> ~165 events total.
SEARCH_DECILES = 10
SEARCH_PER_DECILE_MIN = 30
SEARCH_PER_DECILE_MAX = 35

RANDOM_STATE = 42

# The old sample used RANDOM_STATE=42 on every single run, so
# every optimization run trained on the exact same events. The
# seed below changes on every run instead (random.SystemRandom is
# independent of the random.seed() call further down), while
# staying FIXED for the duration of that run -- every generation
# and every candidate within one run still sees the identical
# sample, which is required for a fair R² comparison between
# candidates. Across different runs, the search sample varies.
SEARCH_SAMPLE_SEED = random.SystemRandom().randint(0, 2**31 - 1)

# Fixed size of the held-out validation batch drawn from the
# validation quarter. The quarter split itself (train vs.
# validation quarter) is unchanged.
VALIDATION_SAMPLE_SIZE = 350


# ------------------------------------------------------------
# Prediction API
# ------------------------------------------------------------

# The Mistral key on this account is limited to
# MISTRAL_MAX_REQUESTS_PER_MINUTE requests/minute, shared across
# BOTH prediction calls and optimizer/candidate-generation calls.
# A global rate limiter (see RATE LIMITING section below)
# serializes every API call and guarantees that limit is never
# exceeded. Because of that, higher concurrency no longer buys
# throughput -- it's kept at 1 so threads don't pile up uselessly
# behind the limiter.
PREDICTION_CONCURRENCY = 1

MAX_RETRIES = 5
BASE_SLEEP = 8

PREDICTION_MAX_TOKENS = 32

MISTRAL_MAX_REQUESTS_PER_MINUTE = 120


# ------------------------------------------------------------
# Optimizer API
# ------------------------------------------------------------

OPTIMIZER_MAX_TOKENS = 5000

OPTIMIZER_MAX_PROMPT_LENGTH = 30000


# ------------------------------------------------------------
# Full evaluation
# ------------------------------------------------------------

# Full optimization dataset is evaluated ONLY after the
# evolutionary search has finished.
#
# Validation is NEVER used during prompt selection.


# ============================================================
# RANDOMNESS
# ============================================================

random.seed(RANDOM_STATE)


# ============================================================
# RATE LIMITING
# ============================================================

# Minimum spacing between any two consecutive Mistral API calls.
# 60 / MISTRAL_MAX_REQUESTS_PER_MINUTE is the theoretical minimum
# spacing (15s at 4/min); +1s is a safety margin against network
# jitter and clock imprecision so the account limit is never
# brushed right at the boundary.
_MIN_SECONDS_BETWEEN_CALLS = 2.2


class RateLimiter:
    """
    Thread-safe limiter shared by EVERY Mistral API call: both
    prediction calls (which run in worker threads via
    asyncio.to_thread) and optimizer/candidate-generation calls
    (which run on the main thread), since they draw on the same
    account-wide per-minute quota.

    Guarantees at least `min_interval` seconds between any two
    consecutive calls, regardless of how many threads try to call
    at once -- callers simply queue up on the lock and are
    released one at a time, evenly spaced. This -- not
    PREDICTION_CONCURRENCY -- is what actually prevents 429 /
    rate-limit errors.
    """

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            sleep_for = self.min_interval - (now - self._last_call)

            if sleep_for > 0:
                time.sleep(sleep_for)

            self._last_call = time.monotonic()


RATE_LIMITER = RateLimiter(_MIN_SECONDS_BETWEEN_CALLS)


def _looks_like_rate_limit_error(exc: Exception) -> bool:
    """
    Best-effort detection of a 429 / rate-limit response, so
    retries can back off for a full minute instead of the normal
    (much shorter) exponential backoff used for other errors.
    """

    text = str(exc).lower()

    return (
        "429" in text
        or "rate limit" in text
        or "rate_limit" in text
        or "too many requests" in text
    )


# ============================================================
# DATA LOADING
# ============================================================

REQUIRED_COLUMNS = {
    "event_id",
    "quarter",
    "facts_text",
    "target_percentile_quarter",
    "earnings_surprise_wins",
}


def load_dataset() -> pd.DataFrame:
    print(f"Loading dataset from: {DATA_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Clean basic fields
    # --------------------------------------------------------

    df["event_id"] = df["event_id"].astype(str)
    df["quarter"] = df["quarter"].astype(str)
    df["facts_text"] = df["facts_text"].fillna("").astype(str)

    df["target_percentile_quarter"] = pd.to_numeric(
        df["target_percentile_quarter"],
        errors="coerce",
    )

    df["earnings_surprise_wins"] = pd.to_numeric(
        df["earnings_surprise_wins"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Remove rows without target
    # --------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=[
            "target_percentile_quarter",
            "earnings_surprise_wins",
        ]
    ).copy()

    print(
        f"Removed {before - len(df)} rows with missing target/surprise."
    )

    # --------------------------------------------------------
    # Calculate cross-sectional surprise percentile
    #
    # This is calculated INSIDE each quarter.
    # --------------------------------------------------------

    df["surprise_percentile_quarter"] = (
        df.groupby("quarter")["earnings_surprise_wins"]
        .rank(pct=True)
    )

    # --------------------------------------------------------
    # Keep valid target range
    # --------------------------------------------------------

    df = df[
        df["target_percentile_quarter"].between(0, 1)
        & df["surprise_percentile_quarter"].between(0, 1)
    ].copy()

    df = df.reset_index(drop=True)

    print(f"Dataset size: {len(df)}")
    print(
        f"Quarters: {df['quarter'].nunique()}"
    )

    return df


# ============================================================
# SEARCH SAMPLE (DECILE-STRATIFIED, INCONSISTENCY-WEIGHTED)
# ============================================================

def build_decile_stratified_sample(
    df: pd.DataFrame,
    n_deciles: int = SEARCH_DECILES,
    per_decile_min: int = SEARCH_PER_DECILE_MIN,
    per_decile_max: int = SEARCH_PER_DECILE_MAX,
    seed: int = SEARCH_SAMPLE_SEED,
) -> pd.DataFrame:
    """
    Build the SEARCH/TRAIN sample used during evolutionary search.

    Stratifies by DECILE of target_percentile_quarter so the
    sample covers the full outcome range (not just whatever a
    given quarter happens to contain), and within each decile
    prefers "inconsistent" events -- ones where
    surprise_percentile_quarter poorly explains
    target_percentile_quarter -- since those carry the most
    incremental signal for the prompt to learn from.

    The sample is FIXED for the duration of one run (built once
    in main() and reused across every generation/candidate, which
    is required for a fair R² comparison), but the seed differs
    between runs, so repeated optimization runs are exposed to
    different, more varied data over time.
    """

    required = {
        "target_percentile_quarter",
        "surprise_percentile_quarter",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Cannot build decile-stratified sample: "
            f"missing columns {sorted(missing)}"
        )

    rng = random.Random(seed)

    work = df.copy()

    # How poorly surprise explains the target for this event.
    # Higher = more "inconsistent" = more informative for search.
    work["_inconsistency"] = (
        work["target_percentile_quarter"]
        - work["surprise_percentile_quarter"]
    ).abs()

    try:
        work["_decile"] = pd.qcut(
            work["target_percentile_quarter"],
            q=n_deciles,
            labels=False,
            duplicates="drop",
        )
    except ValueError as exc:
        raise ValueError(
            "Could not compute deciles of "
            "'target_percentile_quarter'."
        ) from exc

    print(
        f"\nBuilding decile-stratified SEARCH sample "
        f"(seed={seed}):"
    )

    samples = []

    deciles = sorted(work["_decile"].dropna().unique())

    if not deciles:
        raise ValueError(
            "No deciles found for search sample."
        )

    for decile in deciles:
        group = work[work["_decile"] == decile]

        if group.empty:
            continue

        n_target = rng.randint(per_decile_min, per_decile_max)
        n = min(n_target, len(group))

        weights = group["_inconsistency"] + 1e-6

        if weights.isna().any() or weights.sum() <= 0:
            sampled = group.sample(
                n=n,
                random_state=rng.randint(0, 2**31 - 1),
            )
        else:
            # Weighted-random draw: bias toward inconsistent
            # events without deterministically taking only the
            # top-N, so the sample stays random and doesn't
            # collapse onto a handful of extreme outlier rows.
            sampled = group.sample(
                n=n,
                weights=weights,
                random_state=rng.randint(0, 2**31 - 1),
            )

        samples.append(sampled)

        print(
            f"  decile {int(decile)}: {n}/{len(group)} "
            f"(target n={n_target})"
        )

    if not samples:
        raise ValueError(
            "Could not construct decile-stratified search sample."
        )

    sample = pd.concat(samples, ignore_index=True)

    sample = sample.drop(columns=["_inconsistency", "_decile"])

    # Shuffle final order while keeping composition fixed.
    sample = sample.sample(
        frac=1,
        random_state=rng.randint(0, 2**31 - 1),
    ).reset_index(drop=True)

    print(f"\nSearch sample size: {len(sample)} events")

    print(
        sample["quarter"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    return sample


# ============================================================
# VALIDATION SAMPLE
# ============================================================

def build_validation_sample(
    df: pd.DataFrame,
    n: int = VALIDATION_SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Draw a fixed-size (default 100) random sample from the
    validation quarter.

    Uses the fixed RANDOM_STATE (not the per-run search seed) so
    the held-out validation batch stays the same, comparable set
    across successive optimization runs.
    """

    if df.empty:
        raise ValueError(
            "Validation dataframe is empty."
        )

    n = min(n, len(df))

    sample = df.sample(
        n=n,
        random_state=random_state,
    ).reset_index(drop=True)

    print(
        f"\nValidation sample: {len(sample)}/{len(df)} events "
        f"(quarter(s): {sorted(sample['quarter'].unique())})"
    )

    return sample


# ============================================================
# R² CALCULATION
# ============================================================

def calculate_r2(
    sample: pd.DataFrame,
    predictions: pd.Series,
) -> tuple[float, int]:
    """
    Calculate the same OLS R² logic used by the original
    prediction module:

        target ~ prediction + surprise_percentile_quarter
    """

    required = [
        "target_percentile_quarter",
        "surprise_percentile_quarter",
    ]

    for col in required:
        if col not in sample.columns:
            raise ValueError(
                f"Column '{col}' does not exist."
            )

    evaluation = sample[
        required
    ].copy()

    evaluation["prediction"] = pd.to_numeric(
        predictions,
        errors="coerce",
    )

    evaluation = evaluation.dropna()

    if len(evaluation) < 10:
        return float("nan"), len(evaluation)

    if evaluation["prediction"].nunique() < 2:
        return float("nan"), len(evaluation)

    X = sm.add_constant(
        evaluation[
            [
                "prediction",
                "surprise_percentile_quarter",
            ]
        ]
    )

    y = evaluation["target_percentile_quarter"]

    try:
        model = sm.OLS(y, X).fit()
        return float(model.rsquared), len(evaluation)
    except Exception:
        return float("nan"), len(evaluation)


# ============================================================
# PREDICTION PARSING
# ============================================================

def parse_prediction(text: str) -> float:
    """
    Extract a number in [0, 1] from model output.

    The prediction prompt asks for ONLY a number, but the
    parser is deliberately tolerant.
    """

    if not text:
        return 0.5

    text = text.strip()

    # First try a normal decimal between 0 and 1.
    match = re.search(
        r"(?<![\d.])(?:0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])",
        text,
    )

    if match:
        value = float(match.group())
        return max(0.0, min(1.0, value))

    # Fallback: any number.
    match = re.search(
        r"-?\d+(?:\.\d+)?",
        text,
    )

    if match:
        try:
            value = float(match.group())

            if value > 1 and value <= 100:
                value /= 100.0

            return max(0.0, min(1.0, value))

        except ValueError:
            pass

    return 0.5


# ============================================================
# MISTRAL PREDICTION
# ============================================================

def make_prediction_prompt(
    prompt_template: str,
    facts_text: str,
) -> str:
    try:
        return prompt_template.format(
            facts=facts_text
        )
    except Exception as exc:
        raise ValueError(
            "Failed to format BASE_PROMPT with facts. "
            "Make sure the prompt contains '{facts}'."
        ) from exc


def synchronous_prediction(
    prompt_template: str,
    facts_text: str,
) -> dict:
    """
    One synchronous Mistral prediction call.

    This function is executed in a worker thread by asyncio.
    """

    prompt = make_prediction_prompt(
        prompt_template,
        facts_text,
    )

    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            # Blocks (across ALL threads) until it is safe to
            # make another request without exceeding the
            # account's per-minute limit.
            RATE_LIMITER.wait()

            response = client.chat.complete(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                max_tokens=PREDICTION_MAX_TOKENS,
            )

            text = ""

            if response.choices:
                message = response.choices[0].message

                if message is not None:
                    content = message.content

                    if isinstance(content, str):
                        text = content

                    elif content is not None:
                        text = str(content)

            prediction = parse_prediction(text)

            return {
                "prediction": prediction,
                "raw_text": text,
                "error": None,
            }

        except Exception as exc:
            last_error = str(exc)

            if attempt < MAX_RETRIES - 1:

                if _looks_like_rate_limit_error(exc):
                    # Extra-safe backoff specifically for
                    # rate-limit errors: wait a full minute (plus
                    # jitter) so the per-minute quota has
                    # definitely reset, even if something outside
                    # this script also used the same key.
                    sleep_time = 60.0 + random.uniform(0, 5)
                else:
                    sleep_time = BASE_SLEEP * (
                        2 ** attempt
                    )

                    # Small jitter prevents synchronized retries.
                    sleep_time += random.uniform(0, 2)

                time.sleep(sleep_time)

    return {
        "prediction": 0.5,
        "raw_text": None,
        "error": last_error,
    }


# ============================================================
# ASYNC PREDICTION
# ============================================================

async def predict_sample(
    prompt_template: str,
    sample: pd.DataFrame,
    concurrency: int = PREDICTION_CONCURRENCY,
) -> pd.DataFrame:
    """
    Evaluate one prompt on one sample.

    The order of the returned dataframe matches `sample`.
    """

    if "facts_text" not in sample.columns:
        raise ValueError(
            "'facts_text' column is missing."
        )

    semaphore = asyncio.Semaphore(
        max(1, concurrency)
    )

    results = [None] * len(sample)

    async def worker(
        index: int,
        event_id: str,
        facts_text: str,
    ):
        async with semaphore:

            result = await asyncio.to_thread(
                synchronous_prediction,
                prompt_template,
                facts_text,
            )

            result["event_id"] = event_id
            results[index] = result

    tasks = []

    for i, (_, row) in enumerate(
        sample.iterrows()
    ):
        tasks.append(
            worker(
                i,
                str(row["event_id"]),
                str(row["facts_text"]),
            )
        )

    await asyncio.gather(*tasks)

    result_df = pd.DataFrame(results)

    # Safety check.
    if len(result_df) != len(sample):
        raise RuntimeError(
            "Prediction result length does not match sample."
        )

    return result_df


# ============================================================
# PROMPT EVALUATION
# ============================================================

async def evaluate_prompt(
    prompt: str,
    sample: pd.DataFrame,
    save_predictions: bool = False,
    label: Optional[str] = None,
) -> dict:
    """
    Run a prompt over the sample and calculate R².
    """

    predictions = await predict_sample(
        prompt,
        sample,
    )

    merged = sample[
        [
            "event_id",
            "quarter",
            "target_percentile_quarter",
            "surprise_percentile_quarter",
        ]
    ].merge(
        predictions[
            [
                "event_id",
                "prediction",
                "error",
            ]
        ],
        on="event_id",
        how="left",
    )

    r2, n_valid = calculate_r2(
        merged,
        merged["prediction"],
    )

    n_errors = int(
        merged["error"].notna().sum()
    )

    n_fallback = int(
        (
            merged["prediction"] == 0.5
        ).sum()
    )

    result = {
        "r2": r2,
        "n_events": len(sample),
        "n_valid": n_valid,
        "n_errors": n_errors,
        "n_fallback": n_fallback,
        "predictions": merged,
    }

    if save_predictions:
        suffix = label or "evaluation"

        prediction_path = (
            OUTPUT_DIR
            / f"{suffix}_predictions.csv"
        )

        merged.to_csv(
            prediction_path,
            index=False,
        )

        print(
            f"Saved predictions: {prediction_path}"
        )

    return result


# ============================================================
# OPTIMIZER PROMPT
# ============================================================

OPTIMIZER_SYSTEM_PROMPT = r"""
You are the creator of an evolving financial prediction prompt.

Your task is to improve a prompt used by an LLM to predict the cross-sectional percentile of an unexpected stock return following an earnings call.

The objective is NOT to write an explanation of the prompt.
The objective is to discover a materially better prediction strategy and encode it directly into the generated prompt.

You will receive:
1. The current prediction prompt.
2. Its measured R².
3. Results from previous generations.
4. Information about the dataset.
5. A small set of diagnostic information.

You must reason like a quantitative researcher.

CORE QUANTITATIVE PRINCIPLES:
- The target is a cross-sectional percentile within the quarter (0.00 to 1.00).
- The evaluation model is approximately: target_percentile_quarter ~ prediction + surprise_percentile_quarter.
- The prompt must extract INCREMENTAL signal complementary to earnings surprise using strictly the provided facts.
- Do not invent missing facts or introduce external market data.
- Universal Baseline Anchor: All predictions must anchor at 0.50 (the unconditional cross-sectional quarterly median) as a starting point before applying incremental deltas.

EVOLUTIONARY & EXPERIMENTAL MANDATE:
- Do not be afraid to radically restructure or completely rewrite candidate prompts.
- Eliminate rigid boundary "cliffs" (where a 1% metric miss causes a 20-percentile score drop) and asymmetric AND/OR logic traps. Experiment with continuous, weighted additive/subtractive scoring models.
- NEVER allow candidate prompts to instruct the model to "output ONLY a number" or "respond with ONLY the final score" as this suppresses Chain-of-Thought (CoT) reasoning and degrades performance.

ANALYTICAL DIRECTIVES TO ENCODE IN GENERATED PROMPTS:
1. Fama-French 12 Industry Contextualization: Instruct the prediction model to first classify the company into one of the 12 Fama-French industry groups. Industry context must be used strictly to gauge signal severity (e.g., margin sensitivity), NOT to shift the baseline starting percentile away from 0.50 (which causes sector-bias hallucinations).
2. Mathematical Safety & Clamping:
   - Require candidate prompts to specify the absolute denominator formula: Delta % = (Current - Prior) / |Prior| for negative or zero base-rate metrics (e.g., net income turning from negative to positive).
   - Enforce explicit score clamping in candidate prompts: Final Prediction = min(1.00, max(0.00, 0.50 + Sum(Deltas))) to prevent mathematical overflow/underflow outside [0.00, 1.00].
3. Logical Coherence & Signal Hierarchy: Instruct the model to detect and reconcile contradictory or mutually exclusive signals using a strict hierarchy (Guidance > Cash Flow Quality > Margins > Segment Dispersion > Balance Sheet/Liquidity).
4. Predictive Incremental Signals: Focus on forward guidance revisions, operating cash flow quality vs. GAAP earnings, sustainable vs. one-time margin drivers, and structural revenue dispersion.

OUTPUT FORMAT:

Return ONLY the complete replacement prompt.

The replacement prompt MUST contain:
{facts}

The replacement prompt MUST instruct the prediction model to perform step-by-step reasoning first, but end its response with the final number on a separate line formatted strictly as:
FINAL_PREDICTION: [number]

Do not wrap the prompt in Markdown fences.
Do not add commentary before or after the prompt.
"""


def build_optimizer_user_prompt(
    current_prompt: str,
    current_r2: float,
    history: list[dict],
    candidate_number: int,
) -> str:
    history_text = ""

    if history:
        recent = history[-10:]

        history_text = "\n".join(
            [
                (
                    f"Generation {item.get('generation')}: "
                    f"candidate={item.get('candidate')}, "
                    f"R²={item.get('r2')}"
                )
                for item in recent
            ]
        )

    if not history_text:
        history_text = "No previous generation results."

    prompt = f"""
CURRENT PROMPT:

{current_prompt[:OPTIMIZER_MAX_PROMPT_LENGTH]}


CURRENT R²:

{current_r2}


RECENT EVOLUTION HISTORY:

{history_text}


THIS IS CANDIDATE #{candidate_number} FOR THE NEXT GENERATION.

Create a new replacement prompt.

Try to improve the actual predictive reasoning rather than merely
rewriting wording.

Consider whether the current prompt:

- extracts incremental information beyond earnings surprise;
- distinguishes positive and negative information correctly;
- uses forward guidance efficiently;
- handles contradictory signals;
- handles one-time earnings effects;
- recognizes magnitude rather than only direction;
- produces useful cross-sectional ranking information;
- avoids systematic compression toward 0.5;
- avoids blindly following management tone;
- uses numbers in the facts appropriately;
- produces stable predictions across different companies and
  quarters.

Return ONLY the complete new prompt.
"""

    return prompt


# ============================================================
# OPTIMIZER RESPONSE PARSING
# ============================================================

def clean_candidate_prompt(text: str) -> Optional[str]:
    if not text:
        return None

    text = text.strip()

    # Remove accidental Markdown fences.
    text = re.sub(
        r"^```(?:text|python)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    text = text.strip()

    if not text:
        return None

    # Candidate must have {facts}.
    if "{facts}" not in text:
        return None

    # Avoid accidentally accepting tiny malformed outputs.
    if len(text) < 100:
        return None

    return text


# ============================================================
# OPTIMIZER CALL
# ============================================================

def generate_candidate_prompt(
    current_prompt: str,
    current_r2: float,
    history: list[dict],
    candidate_number: int,
) -> Optional[str]:

    user_prompt = build_optimizer_user_prompt(
        current_prompt=current_prompt,
        current_r2=current_r2,
        history=history,
        candidate_number=candidate_number,
    )

    last_error = None

    for attempt in range(MAX_RETRIES):

        try:
            # Same shared limiter as prediction calls -- the
            # per-minute quota is one account-wide budget, not
            # one budget per call type.
            RATE_LIMITER.wait()

            response = client.chat.complete(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": OPTIMIZER_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                max_tokens=OPTIMIZER_MAX_TOKENS,
            )

            text = ""

            if response.choices:
                message = response.choices[0].message

                if message is not None:
                    content = message.content

                    if isinstance(content, str):
                        text = content
                    elif content is not None:
                        text = str(content)

            candidate = clean_candidate_prompt(text)

            if candidate is not None:
                return candidate

            last_error = (
                "Optimizer returned malformed prompt."
            )

        except Exception as exc:
            last_error = str(exc)

            if attempt < MAX_RETRIES - 1:

                if _looks_like_rate_limit_error(exc):
                    sleep_time = 60.0 + random.uniform(0, 5)
                else:
                    sleep_time = BASE_SLEEP * (
                        2 ** attempt
                    )

                    sleep_time += random.uniform(0, 2)

                time.sleep(sleep_time)

    print(
        "WARNING: optimizer failed: "
        f"{last_error}"
    )

    return None


# ============================================================
# SAVE PROMPT
# ============================================================

def save_prompt(
    prompt: str,
    path: Path,
):
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            'BASE_PROMPT = """\n'
        )

        f.write(prompt)

        f.write(
            '\n"""\n'
        )


# ============================================================
# STATE
# ============================================================

def save_state(
    generation: int,
    current_prompt: str,
    current_r2: float,
    best_prompt: str,
    best_r2: float,
):
    state = {
        "generation": generation,
        "current_r2": current_r2,
        "best_r2": best_r2,
        "current_prompt": current_prompt,
        "best_prompt": best_prompt,
        "model": MODEL,
        "search_deciles": SEARCH_DECILES,
        "search_per_decile_min": SEARCH_PER_DECILE_MIN,
        "search_per_decile_max": SEARCH_PER_DECILE_MAX,
        "search_sample_seed": SEARCH_SAMPLE_SEED,
        "generations": GENERATIONS,
        "candidates_per_generation": (
            CANDIDATES_PER_GENERATION
        ),
    }

    with STATE_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            state,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# HISTORY
# ============================================================

def append_history(
    row: dict,
):
    row_df = pd.DataFrame([row])

    if HISTORY_PATH.exists():
        row_df.to_csv(
            HISTORY_PATH,
            mode="a",
            header=False,
            index=False,
        )
    else:
        row_df.to_csv(
            HISTORY_PATH,
            index=False,
        )


# ============================================================
# MAIN EVOLUTIONARY SEARCH
# ============================================================

async def evolutionary_search(
    optimization_sample: pd.DataFrame,
) -> tuple[str, float]:
    """
    Evolutionary prompt search.

    Every generation evaluates candidates on EXACTLY the same
    fixed sample.

    This is critical because otherwise R² differences can be
    caused by sample noise rather than prompt improvements.
    """

    current_prompt = BASE_PROMPT

    print("\n" + "=" * 70)
    print("INITIAL BASELINE")
    print("=" * 70)

    baseline = await evaluate_prompt(
        current_prompt,
        optimization_sample,
        save_predictions=True,
        label="generation_0_baseline",
    )

    current_r2 = baseline["r2"]

    if pd.isna(current_r2):
        raise RuntimeError(
            "Baseline R² is NaN. "
            "Check prediction parsing and dataset."
        )

    best_prompt = current_prompt
    best_r2 = current_r2

    print(
        f"\nBaseline R²: {current_r2:.6f}"
    )

    history = []

    append_history(
        {
            "generation": 0,
            "candidate": "baseline",
            "r2": current_r2,
            "n_events": baseline["n_events"],
            "n_valid": baseline["n_valid"],
            "n_errors": baseline["n_errors"],
            "n_fallback": baseline["n_fallback"],
        }
    )

    save_prompt(
        best_prompt,
        BEST_PROMPT_PATH,
    )

    save_state(
        generation=0,
        current_prompt=current_prompt,
        current_r2=current_r2,
        best_prompt=best_prompt,
        best_r2=best_r2,
    )

    # --------------------------------------------------------
    # Generations
    # --------------------------------------------------------

    for generation in range(
        1,
        GENERATIONS + 1,
    ):

        print("\n")
        print("=" * 70)
        print(
            f"GENERATION {generation}/{GENERATIONS}"
        )
        print("=" * 70)

        generation_candidates = []

        # ----------------------------------------------------
        # Generate candidates
        # ----------------------------------------------------

        for candidate_number in range(
            1,
            CANDIDATES_PER_GENERATION + 1,
        ):

            print(
                f"\nGenerating candidate "
                f"{candidate_number}/"
                f"{CANDIDATES_PER_GENERATION}..."
            )

            candidate_prompt = (
                generate_candidate_prompt(
                    current_prompt=current_prompt,
                    current_r2=current_r2,
                    history=history,
                    candidate_number=candidate_number,
                )
            )

            if candidate_prompt is None:
                print(
                    "Candidate generation failed."
                )
                continue

            print(
                f"Candidate length: "
                f"{len(candidate_prompt)} chars"
            )

            # ------------------------------------------------
            # Evaluate candidate
            # ------------------------------------------------

            result = await evaluate_prompt(
                candidate_prompt,
                optimization_sample,
                save_predictions=False,
            )

            candidate_r2 = result["r2"]

            print(
                f"Candidate {candidate_number} "
                f"R² = {candidate_r2:.6f}"
            )

            append_history(
                {
                    "generation": generation,
                    "candidate": candidate_number,
                    "r2": candidate_r2,
                    "n_events": result["n_events"],
                    "n_valid": result["n_valid"],
                    "n_errors": result["n_errors"],
                    "n_fallback": result["n_fallback"],
                }
            )

            generation_candidates.append(
                {
                    "candidate": candidate_number,
                    "prompt": candidate_prompt,
                    "r2": candidate_r2,
                    "result": result,
                }
            )

            history.append(
                {
                    "generation": generation,
                    "candidate": candidate_number,
                    "r2": candidate_r2,
                }
            )

        # ----------------------------------------------------
        # No valid candidates
        # ----------------------------------------------------

        valid_candidates = [
            x
            for x in generation_candidates
            if x["r2"] is not None
            and not pd.isna(x["r2"])
        ]

        if not valid_candidates:
            print(
                "\nNo valid candidates in this generation."
            )

            save_state(
                generation=generation,
                current_prompt=current_prompt,
                current_r2=current_r2,
                best_prompt=best_prompt,
                best_r2=best_r2,
            )

            continue

        # ----------------------------------------------------
        # Find best candidate
        # ----------------------------------------------------

        generation_best = max(
            valid_candidates,
            key=lambda x: x["r2"],
        )

        generation_best_r2 = (
            generation_best["r2"]
        )

        print(
            f"\nBest candidate this generation: "
            f"#{generation_best['candidate']}"
        )

        print(
            f"Generation best R²: "
            f"{generation_best_r2:.6f}"
        )

        print(
            f"Previous R²: "
            f"{current_r2:.6f}"
        )

        # ----------------------------------------------------
        # Accept only if actually better
        # ----------------------------------------------------

        if generation_best_r2 > current_r2:

            improvement = (
                generation_best_r2
                - current_r2
            )

            current_prompt = (
                generation_best["prompt"]
            )

            current_r2 = generation_best_r2

            print(
                f"ACCEPTED "
                f"(+{improvement:.6f})"
            )

            if current_r2 > best_r2:

                best_r2 = current_r2
                best_prompt = current_prompt

                print(
                    f"NEW GLOBAL BEST: "
                    f"{best_r2:.6f}"
                )

                save_prompt(
                    best_prompt,
                    BEST_PROMPT_PATH,
                )

        else:

            print(
                "REJECTED: no improvement."
            )

        # ----------------------------------------------------
        # Save state after EVERY generation
        # ----------------------------------------------------

        save_state(
            generation=generation,
            current_prompt=current_prompt,
            current_r2=current_r2,
            best_prompt=best_prompt,
            best_r2=best_r2,
        )

        print(
            f"\nCurrent R²: {current_r2:.6f}"
        )

        print(
            f"Global best R²: {best_r2:.6f}"
        )

        print(
            f"Best prompt saved to: "
            f"{BEST_PROMPT_PATH}"
        )

    return best_prompt, best_r2


# ============================================================
# FULL EVALUATION
# ============================================================

async def full_evaluation(
    best_prompt: str,
    full_optimization: pd.DataFrame,
    validation: pd.DataFrame,
):
    print("\n")
    print("=" * 70)
    print("FULL OPTIMIZATION EVALUATION")
    print("=" * 70)

    print(
        f"Events: {len(full_optimization)}"
    )

    optimization_result = await evaluate_prompt(
        best_prompt,
        full_optimization,
        save_predictions=True,
        label="FULL_OPTIMIZATION",
    )

    optimization_r2 = optimization_result["r2"]

    print(
        f"\nFULL OPTIMIZATION R²: "
        f"{optimization_r2:.6f}"
    )

    print(
        f"Valid observations: "
        f"{optimization_result['n_valid']}"
    )

    print(
        f"API errors: "
        f"{optimization_result['n_errors']}"
    )

    print(
        f"Fallback 0.5 predictions: "
        f"{optimization_result['n_fallback']}"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    print(
        f"Validation events: {len(validation)}"
    )

    validation_result = await evaluate_prompt(
        best_prompt,
        validation,
        save_predictions=True,
        label="VALIDATION",
    )

    validation_r2 = validation_result["r2"]

    print(
        f"\nVALIDATION R²: "
        f"{validation_r2:.6f}"
    )

    print(
        f"Valid observations: "
        f"{validation_result['n_valid']}"
    )

    print(
        f"API errors: "
        f"{validation_result['n_errors']}"
    )

    print(
        f"Fallback 0.5 predictions: "
        f"{validation_result['n_fallback']}"
    )

    return {
        "optimization_r2": optimization_r2,
        "validation_r2": validation_r2,
        "optimization_n": optimization_result[
            "n_valid"
        ],
        "validation_n": validation_result[
            "n_valid"
        ],
    }


# ============================================================
# MAIN
# ============================================================

async def main():

    print("=" * 70)
    print("PROMPT EVOLUTION")
    print("=" * 70)

    print(
        f"Model: {MODEL}"
    )

    print(
        f"Generations: {GENERATIONS}"
    )

    print(
        f"Candidates / generation: "
        f"{CANDIDATES_PER_GENERATION}"
    )

    print(
        f"Search sample: {SEARCH_DECILES} deciles x "
        f"{SEARCH_PER_DECILE_MIN}-{SEARCH_PER_DECILE_MAX} "
        f"events/decile (seed={SEARCH_SAMPLE_SEED})"
    )

    print(
        f"Validation sample size: {VALIDATION_SAMPLE_SIZE}"
    )

    print(
        f"Prediction concurrency: "
        f"{PREDICTION_CONCURRENCY}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Train / validation split
    #
    # This assumes validation quarter is the final quarter
    # in the dataset, as in your previous setup.
    #
    # If you have an explicit validation quarter, set it below.
    #
    # NOTE: this quarter-based split is unchanged.
    # --------------------------------------------------------

    validation_quarter = "2026Q2"

    if validation_quarter not in set(
        df["quarter"]
    ):
        raise ValueError(
            f"Validation quarter "
            f"'{validation_quarter}' "
            f"does not exist in dataset."
        )

    full_optimization = df[
        df["quarter"] != validation_quarter
    ].copy()

    validation = df[
        df["quarter"] == validation_quarter
    ].copy()

    if full_optimization.empty:
        raise ValueError(
            "Full optimization dataset is empty."
        )

    if validation.empty:
        raise ValueError(
            "Validation dataset is empty."
        )

    print("\nDataset split:")
    print(
        f"Optimization: {len(full_optimization)}"
    )
    print(
        f"Validation:   {len(validation)}"
    )

    # --------------------------------------------------------
    # Build fixed evolutionary SEARCH sample
    # (decile-stratified, inconsistency-weighted, seed changes
    # every run, ~165 events)
    # --------------------------------------------------------

    optimization_sample = (
        build_decile_stratified_sample(
            full_optimization,
            n_deciles=SEARCH_DECILES,
            per_decile_min=SEARCH_PER_DECILE_MIN,
            per_decile_max=SEARCH_PER_DECILE_MAX,
            seed=SEARCH_SAMPLE_SEED,
        )
    )

    # Save exact search sample.
    search_sample_path = (
        OUTPUT_DIR
        / "evolution_search_sample.csv"
    )

    optimization_sample.to_csv(
        search_sample_path,
        index=False,
    )

    print(
        f"\nFixed search sample saved to: "
        f"{search_sample_path}"
    )

    # --------------------------------------------------------
    # Rough runtime estimate given the rate limit
    # --------------------------------------------------------

    estimated_calls = (
        len(optimization_sample)  # generation-0 baseline
        + GENERATIONS * CANDIDATES_PER_GENERATION
        * (1 + len(optimization_sample))  # 1 optimizer call + eval
    )

    estimated_seconds = (
        estimated_calls * _MIN_SECONDS_BETWEEN_CALLS
    )

    print(
        f"\nEstimated API calls for the search phase: "
        f"~{estimated_calls} "
        f"(~{estimated_seconds / 3600:.1f}h at "
        f"{MISTRAL_MAX_REQUESTS_PER_MINUTE} req/min). "
        f"This does not include the final full-optimization "
        f"and validation evaluation at the end."
    )

    # --------------------------------------------------------
    # Evolution
    # --------------------------------------------------------

    best_prompt, search_r2 = (
        await evolutionary_search(
            optimization_sample
        )
    )

    # --------------------------------------------------------
    # Save best prompt
    # --------------------------------------------------------

    save_prompt(
        best_prompt,
        BEST_PROMPT_PATH,
    )

    print("\n")
    print("=" * 70)
    print("EVOLUTION FINISHED")
    print("=" * 70)

    print(
        f"Search-sample best R²: "
        f"{search_r2:.6f}"
    )

    print(
        f"Best prompt: "
        f"{BEST_PROMPT_PATH}"
    )

    # --------------------------------------------------------
    # Validation batch: strictly 100 events from the validation
    # quarter (the quarter split itself is unchanged).
    # --------------------------------------------------------

    validation_sample = build_validation_sample(
        validation,
        n=VALIDATION_SAMPLE_SIZE,
        random_state=RANDOM_STATE,
    )

    validation_sample_path = (
        OUTPUT_DIR
        / "validation_sample.csv"
    )

    validation_sample.to_csv(
        validation_sample_path,
        index=False,
    )

    print(
        f"\nValidation sample saved to: "
        f"{validation_sample_path}"
    )

    # --------------------------------------------------------
    # Full evaluation
    # --------------------------------------------------------

    final_results = await full_evaluation(
        best_prompt=best_prompt,
        full_optimization=full_optimization,
        validation=validation_sample,
    )

    # --------------------------------------------------------
    # Save final summary
    # --------------------------------------------------------

    summary_path = (
        OUTPUT_DIR
        / "final_results.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "model": MODEL,
                "search_sample_size": len(
                    optimization_sample
                ),
                "search_deciles": SEARCH_DECILES,
                "search_per_decile_min": SEARCH_PER_DECILE_MIN,
                "search_per_decile_max": SEARCH_PER_DECILE_MAX,
                "search_sample_seed": SEARCH_SAMPLE_SEED,
                "full_optimization_size": len(
                    full_optimization
                ),
                "validation_size": len(
                    validation_sample
                ),
                "search_r2": search_r2,
                **final_results,
            },
            f,
            indent=2,
        )

    print("\n")
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"Search sample R²:       "
        f"{search_r2:.6f}"
    )

    print(
        f"Full optimization R²:   "
        f"{final_results['optimization_r2']:.6f}"
    )

    print(
        f"Validation R²:          "
        f"{final_results['validation_r2']:.6f}"
    )

    print(
        f"\nFinal prompt: "
        f"{BEST_PROMPT_PATH}"
    )

    print(
        f"Results: "
        f"{summary_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())