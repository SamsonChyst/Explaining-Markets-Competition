import asyncio
import os
import re
from pathlib import Path
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
    raise RuntimeError(
        "MISTRAL_KEY is not set in environment."
    )

client = Mistral(api_key=MISTRAL_KEY)

# Your dataset
DATA_PATH = Path("dataset_cleaned.csv")

# Files created by this preparation step
OUTPUT_DIR = Path("optimization_data")

# Prompt optimization is performed ONLY on these quarters.
TRAIN_QUARTERS = [
    "2025Q4",
    "2026Q1",
]

# This quarter is NEVER used during prompt optimization.
VAL_QUARTER = "2026Q2"

# Number of simultaneous Mistral requests.
CONCURRENCY = 10

# Retry configuration.
MAX_RETRIES = 5
BASE_SLEEP = 12

# Mistral prediction model.
MODEL = os.getenv(
    "MISTRAL_MODEL",
    "mistral-large-latest",
)

# The prediction should be just a number.
MAX_TOKENS = 32


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset() -> pd.DataFrame:
    """
    Load dataset_cleaned.csv and construct the
    cross-sectional earnings-surprise percentile
    within each quarter.
    """

    print(f"Reading dataset from: {DATA_PATH.resolve()}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Loaded {len(df)} rows.")
    print(f"Columns: {list(df.columns)}")

    # These are the columns that MUST exist in the raw CSV.
    required_columns = {
        "event_id",
        "ticker",
        "quarter",
        "event_datetime",
        "facts_text",
        "earnings_surprise_wins",
        "car1_wins",
        "target_percentile_quarter",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{sorted(missing)}"
        )

    df = df.copy()

    # --------------------------------------------------------
    # Basic normalization
    # --------------------------------------------------------

    df["event_id"] = df["event_id"].astype(str)
    df["ticker"] = df["ticker"].astype(str)
    df["quarter"] = df["quarter"].astype(str)

    # Make sure numeric columns are actually numeric.
    numeric_columns = [
        "earnings_surprise_wins",
        "car1_wins",
        "target_percentile_quarter",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Cross-sectional surprise percentile
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # This is NOT calculated across the whole dataset.
    #
    # Each observation competes ONLY against other
    # observations belonging to the same quarter.
    #
    # Example:
    #
    # 2025Q4 -> ranks among 2025Q4 events
    # 2026Q1 -> ranks among 2026Q1 events
    # 2026Q2 -> ranks among 2026Q2 events
    #
    # This is equivalent to the original code:
    #
    # df.groupby("quarter")["earnings_surprise_wins"].rank(pct=True)
    #

    df["surprise_percentile_quarter"] = (
        df.groupby("quarter")[
            "earnings_surprise_wins"
        ]
        .rank(
            pct=True,
        )
    )

    # --------------------------------------------------------
    # Remove observations that cannot participate in OLS.
    # --------------------------------------------------------
    #
    # We do NOT remove rows globally just because some fields
    # are missing. Prediction failures will be handled later.
    #

    print("\nQuarter distribution:")

    print(
        df["quarter"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nMissing values in important columns:"
    )

    print(
        df[
            [
                "facts_text",
                "earnings_surprise_wins",
                "surprise_percentile_quarter",
                "target_percentile_quarter",
            ]
        ]
        .isna()
        .sum()
        .to_string()
    )

    return df


# ============================================================
# BUILD FIXED OPTIMIZATION / VALIDATION DATASETS
# ============================================================

def build_fixed_samples(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create:

        optimization:
            ALL observations from TRAIN_QUARTERS

        validation:
            ALL observations from VAL_QUARTER

    The optimization sample is FIXED.
    It is never randomly resampled between generations.
    """

    optimization = df[
        df["quarter"].isin(TRAIN_QUARTERS)
    ].copy()

    validation = df[
        df["quarter"] == VAL_QUARTER
    ].copy()

    if optimization.empty:
        raise ValueError(
            "Optimization dataset is empty. "
            f"Expected quarters: {TRAIN_QUARTERS}"
        )

    if validation.empty:
        raise ValueError(
            "Validation dataset is empty. "
            f"Expected quarter: {VAL_QUARTER}"
        )

    optimization = optimization.reset_index(
        drop=True
    )

    validation = validation.reset_index(
        drop=True
    )

    return optimization, validation


# ============================================================
# PROMPT CONSTRUCTION
# ============================================================

def build_prediction_prompt(
    facts_text: str,
    base_prompt: str = BASE_PROMPT,
) -> str:
    """
    Insert the transcript facts into BASE_PROMPT.
    """

    return base_prompt.format(
        facts=facts_text
    )


# ============================================================
# PREDICTION PARSER
# ============================================================

_NUMBER_RE = re.compile(
    r"(?<![\d.])"
    r"(?:0(?:\.\d+)?|1(?:\.0+)?)"
    r"(?![\d.])"
)


def parse_prediction(
    text: str,
) -> float | None:
    """
    Parse the model's percentile prediction.

    Expected output:
        0
        0.25
        0.731
        1

    The intended model response is ONLY a number.

    We first try an exact match and then use a conservative
    fallback parser.
    """

    if not text:
        return None

    text = text.strip()

    # --------------------------------------------------------
    # Exact output.
    # --------------------------------------------------------

    exact_match = re.fullmatch(
        r"(?:0(?:\.\d+)?|1(?:\.0+)?)",
        text,
    )

    if exact_match:
        value = float(text)

        if 0.0 <= value <= 1.0:
            return value

        return None

    # --------------------------------------------------------
    # Fallback.
    # --------------------------------------------------------

    matches = _NUMBER_RE.findall(text)

    for match in matches:

        try:
            value = float(match)

        except ValueError:
            continue

        if 0.0 <= value <= 1.0:
            return value

    return None


# ============================================================
# MISTRAL PREDICTION
# ============================================================

async def get_prediction(
    facts_text: str,
    base_prompt: str = BASE_PROMPT,
) -> dict:
    """
    Ask Mistral for a single prediction.

    Returns:
        prediction
        raw_text
        error
    """

    prompt = build_prediction_prompt(
        facts_text=facts_text,
        base_prompt=base_prompt,
    )

    last_error = None

    for attempt in range(MAX_RETRIES):

        try:

            response = client.chat.complete(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                max_tokens=MAX_TOKENS,
            )

            raw_text = (
                response
                .choices[0]
                .message
                .content
                .strip()
            )

            prediction = parse_prediction(
                raw_text
            )

            if prediction is None:

                return {
                    "prediction": None,
                    "raw_text": raw_text,
                    "error": (
                        "Could not parse a valid "
                        "number between 0 and 1."
                    ),
                }

            return {
                "prediction": prediction,
                "raw_text": raw_text,
                "error": None,
            }

        except Exception as exc:

            last_error = str(exc)

            if attempt < MAX_RETRIES - 1:

                sleep_time = (
                    BASE_SLEEP * (attempt + 1)
                )

                await asyncio.sleep(
                    sleep_time
                )

    return {
        "prediction": None,
        "raw_text": None,
        "error": last_error,
    }


# ============================================================
# CONCURRENT PREDICTIONS
# ============================================================

async def run_predictions(
    sample: pd.DataFrame,
    base_prompt: str = BASE_PROMPT,
    concurrency: int = CONCURRENCY,
) -> pd.DataFrame:
    """
    Run Mistral predictions concurrently.

    The order of the returned DataFrame matches the
    order of the input sample.
    """

    semaphore = asyncio.Semaphore(
        concurrency
    )

    results = [None] * len(sample)

    async def worker(
        index: int,
        event_id: str,
        facts_text: str,
    ):

        async with semaphore:

            result = await get_prediction(
                facts_text=facts_text,
                base_prompt=base_prompt,
            )

            result["event_id"] = event_id

            results[index] = result

    tasks = []

    for i, (_, row) in enumerate(
        sample.iterrows()
    ):

        task = worker(
            i,
            row["event_id"],
            row["facts_text"],
        )

        tasks.append(task)

    await asyncio.gather(
        *tasks
    )

    return pd.DataFrame(
        results
    )


# ============================================================
# R² CALCULATION
# ============================================================

def calculate_r2(
    sample: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[
    float | None,
    pd.DataFrame,
]:
    """
    Merge predictions with targets and calculate:

        target_percentile_quarter
            ~ prediction
            + surprise_percentile_quarter

    Returns:

        r2
        diagnostics dataframe
    """

    merged = sample.merge(
        predictions[
            [
                "event_id",
                "prediction",
                "raw_text",
                "error",
            ]
        ],
        on="event_id",
        how="left",
    )

    # --------------------------------------------------------
    # Valid observations
    # --------------------------------------------------------

    valid = merged.dropna(
        subset=[
            "prediction",
            "target_percentile_quarter",
            "surprise_percentile_quarter",
        ]
    ).copy()

    if len(valid) < 10:

        return None, valid

    # --------------------------------------------------------
    # OLS
    # --------------------------------------------------------

    X = sm.add_constant(
        valid[
            [
                "prediction",
                "surprise_percentile_quarter",
            ]
        ]
    )

    y = valid[
        "target_percentile_quarter"
    ]

    model = sm.OLS(
        y,
        X,
    ).fit()

    # --------------------------------------------------------
    # Fitted values and residuals
    # --------------------------------------------------------

    valid["fitted"] = model.predict(
        X
    )

    valid["residual"] = (
        valid[
            "target_percentile_quarter"
        ]
        - valid["fitted"]
    )

    # Store useful regression diagnostics.
    valid["prediction_coefficient"] = (
        model.params.get(
            "prediction",
            float("nan"),
        )
    )

    valid["surprise_coefficient"] = (
        model.params.get(
            "surprise_percentile_quarter",
            float("nan"),
        )
    )

    return (
        float(model.rsquared),
        valid,
    )


# ============================================================
# MAIN
# ============================================================

async def main():

    # --------------------------------------------------------
    # Create output directory.
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("PREPARING PROMPT OPTIMIZATION DATA")
    print("=" * 70)

    # --------------------------------------------------------
    # Load raw dataset.
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = load_dataset()

    # --------------------------------------------------------
    # Build fixed train / validation sets.
    # --------------------------------------------------------

    optimization, validation = (
        build_fixed_samples(df)
    )

    print("\n" + "-" * 70)
    print("DATASET SPLIT")
    print("-" * 70)

    print(
        f"Optimization quarters: "
        f"{TRAIN_QUARTERS}"
    )

    print(
        f"Validation quarter: "
        f"{VAL_QUARTER}"
    )

    print(
        f"\nOptimization observations: "
        f"{len(optimization)}"
    )

    print(
        f"Validation observations:   "
        f"{len(validation)}"
    )

    print(
        "\nOptimization observations by quarter:"
    )

    print(
        optimization[
            "quarter"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nValidation observations by quarter:"
    )

    print(
        validation[
            "quarter"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Save fixed datasets.
    # --------------------------------------------------------

    optimization.to_csv(
        OUTPUT_DIR
        / "optimization_sample.csv",
        index=False,
    )

    validation.to_csv(
        OUTPUT_DIR
        / "validation_sample.csv",
        index=False,
    )

    print(
        "\nSaved fixed optimization sample:"
    )

    print(
        (
            OUTPUT_DIR
            / "optimization_sample.csv"
        ).resolve()
    )

    print(
        "\nSaved validation sample:"
    )

    print(
        (
            OUTPUT_DIR
            / "validation_sample.csv"
        ).resolve()
    )

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RUNNING BASELINE BASE_PROMPT")
    print("=" * 70)

    print(
        f"\nMistral model: {MODEL}"
    )

    print(
        f"Concurrency: {CONCURRENCY}"
    )

    print(
        f"Events to predict: "
        f"{len(optimization)}"
    )

    print(
        "\nThis may take some time..."
    )

    predictions = await run_predictions(
        sample=optimization,
        base_prompt=BASE_PROMPT,
        concurrency=CONCURRENCY,
    )

    # --------------------------------------------------------
    # Save raw predictions.
    # --------------------------------------------------------

    predictions.to_csv(
        OUTPUT_DIR
        / "baseline_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Calculate baseline R².
    # --------------------------------------------------------

    baseline_r2, diagnostics = (
        calculate_r2(
            sample=optimization,
            predictions=predictions,
        )
    )

    if baseline_r2 is None:

        raise RuntimeError(
            "Could not calculate baseline R². "
            "Fewer than 10 valid observations."
        )

    # --------------------------------------------------------
    # Save diagnostics.
    # --------------------------------------------------------

    diagnostics.to_csv(
        OUTPUT_DIR
        / "baseline_diagnostics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save metadata.
    # --------------------------------------------------------

    metadata = pd.DataFrame(
        [
            {
                "model": MODEL,
                "train_quarters": ",".join(
                    TRAIN_QUARTERS
                ),
                "validation_quarter": (
                    VAL_QUARTER
                ),
                "optimization_sample_size": (
                    len(optimization)
                ),
                "validation_sample_size": (
                    len(validation)
                ),
                "baseline_r2": (
                    baseline_r2
                ),
            }
        ]
    )

    metadata.to_csv(
        OUTPUT_DIR
        / "metadata.csv",
        index=False,
    )

    # ========================================================
    # REPORT
    # ========================================================

    n_total = len(
        predictions
    )

    n_valid = int(
        predictions[
            "prediction"
        ]
        .notna()
        .sum()
    )

    n_errors = int(
        predictions[
            "error"
        ]
        .notna()
        .sum()
    )

    n_missing_target = int(
        optimization[
            "target_percentile_quarter"
        ]
        .isna()
        .sum()
    )

    n_missing_surprise = int(
        optimization[
            "surprise_percentile_quarter"
        ]
        .isna()
        .sum()
    )

    print("\n" + "=" * 70)
    print("PREPARATION COMPLETE")
    print("=" * 70)

    print(
        f"\nModel:                    {MODEL}"
    )

    print(
        f"Optimization events:      {n_total}"
    )

    print(
        f"Valid predictions:        {n_valid}"
    )

    print(
        f"Prediction errors:        {n_errors}"
    )

    print(
        f"Missing target:            "
        f"{n_missing_target}"
    )

    print(
        f"Missing surprise percentile:"
        f" {n_missing_surprise}"
    )

    print(
        f"\nBaseline R²:              "
        f"{baseline_r2:.6f}"
    )

    # --------------------------------------------------------
    # Regression coefficients
    # --------------------------------------------------------

    if len(diagnostics) >= 10:

        X = sm.add_constant(
            diagnostics[
                [
                    "prediction",
                    "surprise_percentile_quarter",
                ]
            ]
        )

        y = diagnostics[
            "target_percentile_quarter"
        ]

        model = sm.OLS(
            y,
            X,
        ).fit()

        print(
            "\nBaseline regression:"
        )

        print(
            "target ~ prediction "
            "+ surprise_percentile"
        )

        print(
            f"\nPrediction coefficient: "
            f"{model.params['prediction']:.6f}"
        )

        print(
            f"Surprise coefficient:   "
            f"{model.params['surprise_percentile_quarter']:.6f}"
        )

        print(
            f"R²:                      "
            f"{model.rsquared:.6f}"
        )

    # --------------------------------------------------------
    # Output files.
    # --------------------------------------------------------

    print(
        "\nOutput directory:"
    )

    print(
        OUTPUT_DIR.resolve()
    )

    print(
        "\nCreated files:"
    )

    for path in sorted(
        OUTPUT_DIR.iterdir()
    ):
        print(
            f"  - {path.name}"
        )

    print(
        "\nYou can now run:"
    )

    print(
        "  python prompt_optimization.py"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())