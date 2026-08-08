import asyncio
import re
import pandas as pd
import statsmodels.api as sm
from dotenv import load_dotenv
from anthropic import AsyncAnthropic
from prompts import BASE_PROMPT

load_dotenv()  

DATA_PATH = ("data/processed/all/dataset_cleaned.csv")

TRAIN_QUARTERS = ["2025Q4", "2026Q1"]
VAL_QUARTER = "2026Q2"
CONCURRENCY = 10
MODEL = "claude-sonnet-5"
SAMPLE_SIZE = 100
RANDOM_STATE = 42
INPUT_PRICE_PER_M = 2.0
OUTPUT_PRICE_PER_M = 10.0 


df = pd.read_csv(DATA_PATH)

df["event_datetime"] = pd.to_datetime(df["event_datetime"], utc=True)
df["surprise_percentile_quarter"] = (df.groupby("quarter")["earnings_surprise_wins"].rank(pct=True))


train = df[df["quarter"].isin(TRAIN_QUARTERS)].copy()
val = df[df["quarter"] == VAL_QUARTER].copy()


client = AsyncAnthropic() 

async def get_prediction_with_usage(facts_text: str) -> dict:
    prompt = BASE_PROMPT.format(facts=facts_text)

    try:
        response = await client.messages.create(
            model=MODEL,
            max_tokens=20,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        match = re.search(r"[0-9.]+", text)
        val_pred = max(0.0, min(1.0, float(match.group()))) if match else 0.5

        return {
            "prediction": val_pred,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "raw_text": text,
            "error": None,
        }
    except Exception as exc:
        return {
            "prediction": 0.5,
            "input_tokens": 0,
            "output_tokens": 0,
            "raw_text": None,
            "error": str(exc),
        }


async def run_sample(sample: pd.DataFrame, concurrency: int = CONCURRENCY) -> pd.DataFrame:
    semaphore = asyncio.Semaphore(concurrency)
    results = [None] * len(sample)

    async def worker(i: int, event_id: str, facts: str):
        async with semaphore:
            res = await get_prediction_with_usage(facts)
            res["event_id"] = event_id
            results[i] = res

    tasks = [
        worker(i, row["event_id"], row["facts_text"])
        for i, (_, row) in enumerate(sample.iterrows())
    ]
    await asyncio.gather(*tasks)

    return pd.DataFrame(results)


async def main():
    sample = train.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).copy()

    usage_df = await run_sample(sample)

    usage_df.to_csv("test_sample_100_results.csv", index=False)

    n_errors = usage_df["error"].notna().sum()
    if n_errors > 0:
        print(usage_df[usage_df["error"].notna()][["event_id", "error"]].head())

    n_fallback = (usage_df["prediction"] == 0.5).sum()
    print(f"Prediction 0.5; possible fallback): {n_fallback}/{len(usage_df)}")

    merged = sample.merge(usage_df[["event_id", "prediction"]], on="event_id")
    valid = merged.dropna(subset=[
        "prediction", "target_percentile_quarter", "surprise_percentile_quarter"
    ])

    if len(valid) > 10:
        X = sm.add_constant(valid[["prediction", "surprise_percentile_quarter"]])
        y = valid["target_percentile_quarter"]
        model = sm.OLS(y, X).fit()
        print(f"\nR² {len(valid)} events: {model.rsquared:.4f}")
    else:
        print("\nERR")


if __name__ == "__main__":
    asyncio.run(main())