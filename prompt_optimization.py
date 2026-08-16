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
    raise EnvironmentError("MISTRAL_KEY is not set in the environment.")
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

# ------------------------------------------------------------
# Evolution
# ------------------------------------------------------------
GENERATIONS = 4
CANDIDATES_PER_GENERATION = 2
SEARCH_DECILES = 10
SEARCH_PER_DECILE_MIN = 30
SEARCH_PER_DECILE_MAX = 30
RANDOM_STATE = 67
SEARCH_SAMPLE_SEED = random.SystemRandom().randint(0, 2**31 - 1)
VALIDATION_SAMPLE_SIZE = 350

# ------------------------------------------------------------
# Prediction API
# ------------------------------------------------------------
PREDICTION_CONCURRENCY = 1
MAX_RETRIES = 5
BASE_SLEEP = 8
PREDICTION_MAX_TOKENS = 32
MISTRAL_MAX_REQUESTS_PER_MINUTE = 120

# ------------------------------------------------------------
# Optimizer API
# ------------------------------------------------------------
OPTIMIZER_MAX_TOKENS = 7000
OPTIMIZER_MAX_PROMPT_LENGTH = 30000

# ------------------------------------------------------------
# Skip Baseline Validation
# ------------------------------------------------------------
SKIP_BASELINE_VALIDATION = False
KNOWN_BASELINE_R2 = 0.175846  # Set to your known R²

# ============================================================
# RANDOMNESS
# ============================================================
random.seed(RANDOM_STATE)

# ============================================================
# RATE LIMITING
# ============================================================
_MIN_SECONDS_BETWEEN_CALLS = 2.2

class RateLimiter:
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
    text = str(exc).lower()
    return ("429" in text or "rate limit" in text or "rate_limit" in text or "too many requests" in text)

# ============================================================
# DATA LOADING
# ============================================================
REQUIRED_COLUMNS = {"event_id", "quarter", "facts_text", "target_percentile_quarter", "earnings_surprise_wins"}

def load_dataset() -> pd.DataFrame:
    print(f"Loading dataset from: {DATA_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df["event_id"] = df["event_id"].astype(str)
    df["quarter"] = df["quarter"].astype(str)
    df["facts_text"] = df["facts_text"].fillna("").astype(str)
    df["target_percentile_quarter"] = pd.to_numeric(df["target_percentile_quarter"], errors="coerce")
    df["earnings_surprise_wins"] = pd.to_numeric(df["earnings_surprise_wins"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["target_percentile_quarter", "earnings_surprise_wins"]).copy()
    print(f"Removed {before - len(df)} rows with missing target/surprise.")

    df["surprise_percentile_quarter"] = df.groupby("quarter")["earnings_surprise_wins"].rank(pct=True)
    df = df[df["target_percentile_quarter"].between(0, 1) & df["surprise_percentile_quarter"].between(0, 1)].copy()
    df = df.reset_index(drop=True)
    print(f"Dataset size: {len(df)}")
    print(f"Quarters: {df['quarter'].nunique()}")
    return df

# ============================================================
# SEARCH SAMPLE
# ============================================================
def build_decile_stratified_sample(
    df: pd.DataFrame,
    n_deciles: int = SEARCH_DECILES,
    per_decile_min: int = SEARCH_PER_DECILE_MIN,
    per_decile_max: int = SEARCH_PER_DECILE_MAX,
    seed: int = SEARCH_SAMPLE_SEED,
) -> pd.DataFrame:
    required = {"target_percentile_quarter", "surprise_percentile_quarter"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cannot build decile-stratified sample: missing columns {sorted(missing)}")

    rng = random.Random(seed)
    work = df.copy()
    work["_inconsistency"] = (work["target_percentile_quarter"] - work["surprise_percentile_quarter"]).abs()

    try:
        work["_decile"] = pd.qcut(work["target_percentile_quarter"], q=n_deciles, labels=False, duplicates="drop")
    except ValueError as exc:
        raise ValueError("Could not compute deciles of 'target_percentile_quarter'.") from exc

    print(f"\nBuilding decile-stratified SEARCH sample (seed={seed}):")
    samples = []
    deciles = sorted(work["_decile"].dropna().unique())
    if not deciles:
        raise ValueError("No deciles found for search sample.")

    for decile in deciles:
        group = work[work["_decile"] == decile]
        if group.empty:
            continue
        n_target = rng.randint(per_decile_min, per_decile_max)
        n = min(n_target, len(group))
        weights = group["_inconsistency"] + 1e-6
        if weights.isna().any() or weights.sum() <= 0:
            sampled = group.sample(n=n, random_state=rng.randint(0, 2**31 - 1))
        else:
            sampled = group.sample(n=n, weights=weights, random_state=rng.randint(0, 2**31 - 1))
        samples.append(sampled)
        print(f"  decile {int(decile)}: {n}/{len(group)} (target n={n_target})")

    if not samples:
        raise ValueError("Could not construct decile-stratified search sample.")
    sample = pd.concat(samples, ignore_index=True)
    sample = sample.drop(columns=["_inconsistency", "_decile"])
    sample = sample.sample(frac=1, random_state=rng.randint(0, 2**31 - 1)).reset_index(drop=True)
    print(f"\nSearch sample size: {len(sample)} events")
    print(sample["quarter"].value_counts().sort_index().to_string())
    return sample

# ============================================================
# VALIDATION SAMPLE
# ============================================================
def build_validation_sample(df: pd.DataFrame, n: int = VALIDATION_SAMPLE_SIZE, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Validation dataframe is empty.")
    n = min(n, len(df))
    sample = df.sample(n=n, random_state=random_state).reset_index(drop=True)
    print(f"\nValidation sample: {len(sample)}/{len(df)} events (quarter(s): {sorted(sample['quarter'].unique())})")
    return sample

# ============================================================
# R² CALCULATION
# ============================================================
def calculate_r2(sample: pd.DataFrame, predictions: pd.Series) -> tuple[float, int]:
    required = ["target_percentile_quarter", "surprise_percentile_quarter"]
    for col in required:
        if col not in sample.columns:
            raise ValueError(f"Column '{col}' does not exist.")
    evaluation = sample[required].copy()
    evaluation["prediction"] = pd.to_numeric(predictions, errors="coerce")
    evaluation = evaluation.dropna()
    if len(evaluation) < 10 or evaluation["prediction"].nunique() < 2:
        return float("nan"), len(evaluation)
    X = sm.add_constant(evaluation[["prediction", "surprise_percentile_quarter"]])
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
    if not text:
        return 0.5
    text = text.strip()
    final_pred_match = re.search(r"FINAL_PREDICTION:\s*(0(?:\.\d+)?|1(?:\.0+)?)", text, re.IGNORECASE)
    if final_pred_match:
        return max(0.0, min(1.0, float(final_pred_match.group(1))))
    standalone_match = re.search(r"(?<!\d)(0(?:\.\d+)?|1(?:\.0+)?)(?!\d)", text)
    if standalone_match:
        return max(0.0, min(1.0, float(standalone_match.group(1))))
    return 0.5

# ============================================================
# MISTRAL PREDICTION
# ============================================================
def make_prediction_prompt(prompt_template: str, facts_text: str) -> str:
    try:
        return prompt_template.format(facts=facts_text)
    except Exception as exc:
        raise ValueError("Failed to format BASE_PROMPT with facts. Make sure the prompt contains '{facts}'.") from exc

def synchronous_prediction(prompt_template: str, facts_text: str) -> dict:
    prompt = make_prediction_prompt(prompt_template, facts_text)
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            RATE_LIMITER.wait()
            response = client.chat.complete(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=PREDICTION_MAX_TOKENS,
            )
            text = ""
            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content
                text = content if isinstance(content, str) else str(content) if content is not None else ""
            return {"prediction": parse_prediction(text), "raw_text": text, "error": None}
        except Exception as exc:
            last_error = str(exc)
            if attempt < MAX_RETRIES - 1:
                sleep_time = 60.0 + random.uniform(0, 5) if _looks_like_rate_limit_error(exc) else BASE_SLEEP * (2 ** attempt) + random.uniform(0, 2)
                time.sleep(sleep_time)
    return {"prediction": 0.5, "raw_text": None, "error": last_error}

# ============================================================
# ASYNC PREDICTION
# ============================================================
async def predict_sample(prompt_template: str, sample: pd.DataFrame, concurrency: int = PREDICTION_CONCURRENCY) -> pd.DataFrame:
    if "facts_text" not in sample.columns:
        raise ValueError("'facts_text' column is missing.")
    semaphore = asyncio.Semaphore(max(1, concurrency))
    results = [None] * len(sample)

    async def worker(index: int, event_id: str, facts_text: str):
        async with semaphore:
            result = await asyncio.to_thread(synchronous_prediction, prompt_template, facts_text)
            result["event_id"] = event_id
            results[index] = result

    tasks = [worker(i, str(row["event_id"]), str(row["facts_text"])) for i, (_, row) in enumerate(sample.iterrows())]
    await asyncio.gather(*tasks)
    result_df = pd.DataFrame(results)
    if len(result_df) != len(sample):
        raise RuntimeError("Prediction result length does not match sample.")
    return result_df

# ============================================================
# PROMPT EVALUATION
# ============================================================
async def evaluate_prompt(prompt: str, sample: pd.DataFrame, save_predictions: bool = False, label: Optional[str] = None) -> dict:
    predictions = await predict_sample(prompt, sample)
    merged = sample[["event_id", "quarter", "target_percentile_quarter", "surprise_percentile_quarter"]].merge(
        predictions[["event_id", "prediction", "error"]], on="event_id", how="left"
    )
    r2, n_valid = calculate_r2(merged, merged["prediction"])
    n_errors = int(merged["error"].notna().sum())
    n_fallback = int((merged["prediction"] == 0.5).sum())
    result = {
        "r2": r2, "n_events": len(sample), "n_valid": n_valid,
        "n_errors": n_errors, "n_fallback": n_fallback, "predictions": merged
    }
    if save_predictions:
        suffix = label or "evaluation"
        prediction_path = OUTPUT_DIR / f"{suffix}_predictions.csv"
        merged.to_csv(prediction_path, index=False)
        print(f"Saved predictions: {prediction_path}")
    return result

# ============================================================
# OPTIMIZER PROMPT
# ============================================================
OPTIMIZER_SYSTEM_PROMPT = """
You are the architect of a financial prediction prompt. Your task is to design a prompt that enables an LLM to predict the cross-sectional percentile (0.00-1.00) of a stock's unexpected return following an earnings call, INCREMENTAL TO EARNINGS SURPRISE.

Core Principles:
1. The target is a cross-sectional percentile within the quarter.
2. The evaluation model is: target_percentile_quarter ~ prediction + surprise_percentile_quarter.
3. The prompt must extract incremental signal beyond earnings surprise using ONLY the provided facts.
4. All predictions MUST start at 0.50 and adjust via deltas, clamped to [0.00, 1.00].

Fama-French 12 Industry Benchmarks (Gross Margin %, ROE %):
BusEq(52,0) Chems(35,0) Durbl(27,2) Enrgy(38,3) Hlth(45,-42) Manuf(32,4) Money(55,1) NoDur(39,4) Other(34,-3) Shops(30,5) Telcm(48,2) Utils(38,5)

Signal Priority (STRICT ORDER):
1. Guidance Revisions (highest weight)
2. Cash Flow Dynamics
3. Margin Sustainability
4. Revenue Trends (asymmetric: penalize declines more than growth)
5. Balance Sheet Health (lowest weight)

Required Structure for New Prompts:
1. Assign FF12 industry from {facts} as Step 1.
2. Enforce STRICT signal priority. Higher-priority signals OVERRIDE lower-priority ones.
3. Penalize weak forward language (e.g., "headwinds", "pressure", "slowing", "challenging").
4. Ignore one-time items (e.g., "asset sale", "restructuring").
5. Start at 0.50. Adjust based on priority. Clamp to [0.00, 1.00].
6. Output MUST be ONLY a numeric float (e.g., `0.73`). NO TEXT, NO PREFIXES, NO EXPLANATIONS.

Prohibited:
- Baseline shifting away from 0.50.
- Symmetric treatment of revenue growth/decline.
- Rigid thresholds (e.g., "must have ALL of A, B, C").
- Output formats other than a raw float.

Output Format:
Return ONLY the complete replacement prompt. Do not add commentary, explanations, or formatting.
"""

def build_optimizer_user_prompt(current_prompt: str, current_r2: float, history: list[dict], candidate_number: int) -> str:
    history_text = "\n".join(
        [f"Generation {item.get('generation')}: candidate={item.get('candidate')}, R²={item.get('r2')}"
         for item in (history[-10:] if history else [])]
    ) or "No previous generation results."

    return f"""
CURRENT PROMPT:
{current_prompt}

CURRENT R²:
{current_r2}

RECENT EVOLUTION HISTORY:
{history_text}

THIS IS CANDIDATE #{candidate_number} FOR THE NEXT GENERATION.

Create a new replacement prompt.

Focus on:
- Extracting incremental signal beyond earnings surprise.
- Enforcing STRICT signal priority (Guidance > Cash Flow > Margin > Revenue).
- Asymmetric revenue handling (declines penalized more than growth).
- Penalizing weak forward language.
- Ensuring output is ONLY a numeric float.

Return ONLY the complete new prompt.
"""

# ============================================================
# OPTIMIZER RESPONSE PARSING (FIXED REGEX)
# ============================================================
def clean_candidate_prompt(text: str) -> Optional[str]:
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:text|python)?\s*", "", text, flags=re.IGNORECASE)  # FIXED: `(?\:` → `(?:`
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    if not text or "{facts}" not in text or len(text) < 100:
        return None
    return text

# ============================================================
# OPTIMIZER CALL
# ============================================================
def generate_candidate_prompt(current_prompt: str, current_r2: float, history: list[dict], candidate_number: int) -> Optional[str]:
    user_prompt = build_optimizer_user_prompt(current_prompt, current_r2, history, candidate_number)
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            RATE_LIMITER.wait()
            response = client.chat.complete(
                model=MODEL,
                messages=[
                    {"role": "system", "content": OPTIMIZER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=OPTIMIZER_MAX_TOKENS,
            )
            text = ""
            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content
                text = content if isinstance(content, str) else str(content) if content is not None else ""
            candidate = clean_candidate_prompt(text)
            if candidate is not None:
                return candidate
            last_error = "Optimizer returned malformed prompt."
        except Exception as exc:
            last_error = str(exc)
            if attempt < MAX_RETRIES - 1:
                sleep_time = 60.0 + random.uniform(0, 5) if _looks_like_rate_limit_error(exc) else BASE_SLEEP * (2 ** attempt) + random.uniform(0, 2)
                time.sleep(sleep_time)
    print(f"WARNING: optimizer failed: {last_error}")
    return None

# ============================================================
# SAVE PROMPT
# ============================================================
def save_prompt(prompt: str, path: Path):
    with path.open("w", encoding="utf-8") as f:
        f.write('BASE_PROMPT = """\n' + prompt + '\n"""')

# ============================================================
# STATE
# ============================================================
def save_state(generation: int, current_prompt: str, current_r2: float, best_prompt: str, best_r2: float):
    state = {
        "generation": generation, "current_r2": current_r2, "best_r2": best_r2,
        "current_prompt": current_prompt, "best_prompt": best_prompt,
        "model": MODEL, "search_deciles": SEARCH_DECILES,
        "search_per_decile_min": SEARCH_PER_DECILE_MIN, "search_per_decile_max": SEARCH_PER_DECILE_MAX,
        "search_sample_seed": SEARCH_SAMPLE_SEED, "generations": GENERATIONS,
        "candidates_per_generation": CANDIDATES_PER_GENERATION,
    }
    with STATE_PATH.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

# ============================================================
# HISTORY
# ============================================================
def append_history(row: dict):
    row_df = pd.DataFrame([row])
    if HISTORY_PATH.exists():
        row_df.to_csv(HISTORY_PATH, mode="a", header=False, index=False)
    else:
        row_df.to_csv(HISTORY_PATH, index=False)

# ============================================================
# MAIN EVOLUTIONARY SEARCH (WITH SKIP BASELINE)
# ============================================================
async def evolutionary_search(optimization_sample: pd.DataFrame) -> tuple[str, float]:
    current_prompt = BASE_PROMPT
    print("\n" + "=" * 70)
    print("INITIAL BASELINE")
    print("=" * 70)

    if SKIP_BASELINE_VALIDATION:
        current_r2 = KNOWN_BASELINE_R2
        print(f"SKIPPED baseline validation. Using known R²: {current_r2:.6f}")
        baseline = {
            "r2": current_r2, "n_events": len(optimization_sample),
            "n_valid": len(optimization_sample), "n_errors": 0, "n_fallback": 0, "predictions": None
        }
    else:
        baseline = await evaluate_prompt(current_prompt, optimization_sample, save_predictions=True, label="generation_0_baseline")
        current_r2 = baseline["r2"]
        if pd.isna(current_r2):
            raise RuntimeError("Baseline R² is NaN. Check prediction parsing and dataset.")

    best_prompt, best_r2 = current_prompt, current_r2
    print(f"\nBaseline R²: {current_r2:.6f}")
    history = []
    append_history({
        "generation": 0, "candidate": "baseline", "r2": current_r2,
        "n_events": baseline["n_events"], "n_valid": baseline["n_valid"],
        "n_errors": baseline["n_errors"], "n_fallback": baseline["n_fallback"],
    })
    save_prompt(best_prompt, BEST_PROMPT_PATH)
    save_state(generation=0, current_prompt=current_prompt, current_r2=current_r2, best_prompt=best_prompt, best_r2=best_r2)

    for generation in range(1, GENERATIONS + 1):
        print("\n" + "=" * 70)
        print(f"GENERATION {generation}/{GENERATIONS}")
        print("=" * 70)
        generation_candidates = []
        for candidate_number in range(1, CANDIDATES_PER_GENERATION + 1):
            print(f"\nGenerating candidate {candidate_number}/{CANDIDATES_PER_GENERATION}...")
            candidate_prompt = generate_candidate_prompt(current_prompt, current_r2, history, candidate_number)
            if candidate_prompt is None:
                print("Candidate generation failed.")
                continue
            print(f"Candidate length: {len(candidate_prompt)} chars")
            result = await evaluate_prompt(candidate_prompt, optimization_sample, save_predictions=False)
            candidate_r2 = result["r2"]
            print(f"Candidate {candidate_number} R² = {candidate_r2:.6f}")
            append_history({
                "generation": generation, "candidate": candidate_number, "r2": candidate_r2,
                "n_events": result["n_events"], "n_valid": result["n_valid"],
                "n_errors": result["n_errors"], "n_fallback": result["n_fallback"],
            })
            generation_candidates.append({"candidate": candidate_number, "prompt": candidate_prompt, "r2": candidate_r2, "result": result})
            history.append({"generation": generation, "candidate": candidate_number, "r2": candidate_r2})

        valid_candidates = [x for x in generation_candidates if x["r2"] is not None and not pd.isna(x["r2"])]
        if not valid_candidates:
            print("\nNo valid candidates in this generation.")
            save_state(generation=generation, current_prompt=current_prompt, current_r2=current_r2, best_prompt=best_prompt, best_r2=best_r2)
            continue

        generation_best = max(valid_candidates, key=lambda x: x["r2"])
        generation_best_r2 = generation_best["r2"]
        print(f"\nBest candidate this generation: #{generation_best['candidate']}")
        print(f"Generation best R²: {generation_best_r2:.6f} (prev: {current_r2:.6f})")

        if generation_best_r2 > current_r2:
            improvement = generation_best_r2 - current_r2
            current_prompt = generation_best["prompt"]
            current_r2 = generation_best_r2
            print(f"ACCEPTED (+{improvement:.6f})")
            if current_r2 > best_r2:
                best_r2 = current_r2
                best_prompt = current_prompt
                print(f"NEW GLOBAL BEST: {best_r2:.6f}")
                save_prompt(best_prompt, BEST_PROMPT_PATH)
        else:
            print("REJECTED: no improvement.")

        save_state(generation=generation, current_prompt=current_prompt, current_r2=current_r2, best_prompt=best_prompt, best_r2=best_r2)
        print(f"\nCurrent R²: {current_r2:.6f} | Global best R²: {best_r2:.6f}")

    return best_prompt, best_r2

# ============================================================
# FULL EVALUATION
# ============================================================
async def full_evaluation(best_prompt: str, full_optimization: pd.DataFrame, validation: pd.DataFrame):
    print("\n" + "=" * 70)
    print("FULL OPTIMIZATION EVALUATION")
    print("=" * 70)
    print(f"Events: {len(full_optimization)}")
    optimization_result = await evaluate_prompt(best_prompt, full_optimization, save_predictions=True, label="FULL_OPTIMIZATION")
    optimization_r2 = optimization_result["r2"]
    print(f"R²: {optimization_r2:.6f} | Valid: {optimization_result['n_valid']} | Errors: {optimization_result['n_errors']} | Fallbacks: {optimization_result['n_fallback']}")

    print("\n" + "=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)
    print(f"Events: {len(validation)}")
    validation_result = await evaluate_prompt(best_prompt, validation, save_predictions=True, label="VALIDATION")
    validation_r2 = validation_result["r2"]
    print(f"R²: {validation_r2:.6f} | Valid: {validation_result['n_valid']} | Errors: {validation_result['n_errors']} | Fallbacks: {validation_result['n_fallback']}")

    return {
        "optimization_r2": optimization_r2, "validation_r2": validation_r2,
        "optimization_n": optimization_result["n_valid"], "validation_n": validation_result["n_valid"],
    }

# ============================================================
# MAIN
# ============================================================
async def main():
    print("=" * 70)
    print("PROMPT EVOLUTION")
    print("=" * 70)
    print(f"Model: {MODEL} | Generations: {GENERATIONS} | Candidates/gen: {CANDIDATES_PER_GENERATION}")
    print(f"Skip baseline: {SKIP_BASELINE_VALIDATION} (R²={KNOWN_BASELINE_R2:.6f})" if SKIP_BASELINE_VALIDATION else "Skip baseline: False")

    df = load_dataset()
    validation_quarter = "2026Q2"
    if validation_quarter not in set(df["quarter"]):
        raise ValueError(f"Validation quarter '{validation_quarter}' not in dataset.")
    full_optimization = df[df["quarter"] != validation_quarter].copy()
    validation = df[df["quarter"] == validation_quarter].copy()
    if full_optimization.empty or validation.empty:
        raise ValueError("Optimization or validation dataset is empty.")
    print(f"\nDataset split: Optimization={len(full_optimization)} | Validation={len(validation)}")

    optimization_sample = build_decile_stratified_sample(
        full_optimization, n_deciles=SEARCH_DECILES,
        per_decile_min=SEARCH_PER_DECILE_MIN, per_decile_max=SEARCH_PER_DECILE_MAX,
        seed=SEARCH_SAMPLE_SEED,
    )
    optimization_sample.to_csv(OUTPUT_DIR / "evolution_search_sample.csv", index=False)
    print(f"\nSearch sample: {len(optimization_sample)} events (seed={SEARCH_SAMPLE_SEED})")

    estimated_calls = len(optimization_sample) + GENERATIONS * CANDIDATES_PER_GENERATION * (1 + len(optimization_sample))
    print(f"Estimated API calls: ~{estimated_calls} (~{estimated_calls * _MIN_SECONDS_BETWEEN_CALLS / 3600:.1f}h)")

    best_prompt, search_r2 = await evolutionary_search(optimization_sample)
    save_prompt(best_prompt, BEST_PROMPT_PATH)
    print("\n" + "=" * 70)
    print(f"EVOLUTION FINISHED | Search R²: {search_r2:.6f}")
    print("=" * 70)

    validation_sample = build_validation_sample(validation, n=VALIDATION_SAMPLE_SIZE, random_state=RANDOM_STATE)
    validation_sample.to_csv(OUTPUT_DIR / "validation_sample.csv", index=False)
    final_results = await full_evaluation(best_prompt, full_optimization, validation_sample)

    with (OUTPUT_DIR / "final_results.json").open("w", encoding="utf-8") as f:
        json.dump({
            "model": MODEL, "search_sample_size": len(optimization_sample),
            "validation_size": len(validation_sample), "search_r2": search_r2,
            **final_results,
        }, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Search R²: {search_r2:.6f} | Optimization R²: {final_results['optimization_r2']:.6f} | Validation R²: {final_results['validation_r2']:.6f}")
    print(f"Prompt: {BEST_PROMPT_PATH} | Results: {OUTPUT_DIR / 'final_results.json'}")

if __name__ == "__main__":
    asyncio.run(main())
