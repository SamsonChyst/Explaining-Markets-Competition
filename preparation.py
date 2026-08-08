"""
run_full_train.py — полный прогон train с baseline-промптом + чекпоинты
"""

import asyncio
import json
import re
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from dotenv import load_dotenv
from anthropic import AsyncAnthropic

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════
# КОНФИГ (без SAMPLE_SIZE — гоняем весь train)
# ═══════════════════════════════════════════════════════════════════════

DATA_PATH = ("/Users/bogdanlevchenko/Desktop/Optiver/Preparation/data/processed/all/dataset_cleaned.csv")

TRAIN_QUARTERS = ["2025Q4", "2026Q1"]
VAL_QUARTER = "2026Q2"
CONCURRENCY = 10
MODEL = "claude-sonnet-5"
CHECKPOINT_PATH = Path("train_full_baseline.jsonl")

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


# ═══════════════════════════════════════════════════════════════════════
# ЗАГРУЗКА
# ═══════════════════════════════════════════════════════════════════════

df = pd.read_csv(DATA_PATH)
df["event_datetime"] = pd.to_datetime(df["event_datetime"], utc=True)

df["surprise_percentile_quarter"] = (
    df.groupby("quarter")["earnings_surprise_wins"].rank(pct=True)
)

train = df[df["quarter"].isin(TRAIN_QUARTERS)].copy()
val = df[df["quarter"] == VAL_QUARTER].copy()

assert train["event_datetime"].max() < val["event_datetime"].min(), \
    "УТЕЧКА ДАННЫХ: train пересекается с validation"

print(f"Train: {len(train)} | Val: {len(val)}")

client = AsyncAnthropic()


# ═══════════════════════════════════════════════════════════════════════
# ПРЕДСКАЗАНИЕ
# ═══════════════════════════════════════════════════════════════════════

async def get_prediction_with_usage(facts_text: str) -> dict:
    prompt = BASE_PROMPT.format(facts=facts_text)
    try:
        response = await client.messages.create(
            model=MODEL, max_tokens=20,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        match = re.search(r"[0-9.]+", text)
        val_pred = max(0.0, min(1.0, float(match.group()))) if match else 0.5
        return {
            "prediction": val_pred,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "error": None,
        }
    except Exception as exc:
        return {"prediction": 0.5, "input_tokens": 0, "output_tokens": 0, "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════
# ЧЕКПОИНТЫ — критично на 4000+ вызовах
# ═══════════════════════════════════════════════════════════════════════

def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    done = {}
    with open(path, "r") as f:
        for line in f:
            rec = json.loads(line)
            done[rec["event_id"]] = rec
    return done


def append_checkpoint(path: Path, event_id: str, record: dict):
    with open(path, "a") as f:
        f.write(json.dumps({"event_id": event_id, **record}) + "\n")


# ═══════════════════════════════════════════════════════════════════════
# БАТЧ-ОБРАБОТКА С ВОЗОБНОВЛЕНИЕМ
# ═══════════════════════════════════════════════════════════════════════

async def run_full(data: pd.DataFrame, checkpoint_path: Path,
                    concurrency: int = CONCURRENCY) -> pd.DataFrame:
    done = load_checkpoint(checkpoint_path)
    print(f"Уже готово (из прошлых запусков): {len(done)}")

    todo = data[~data["event_id"].isin(done.keys())]
    print(f"Осталось обработать: {len(todo)}")

    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    total = len(todo)

    async def worker(event_id: str, facts: str):
        nonlocal completed
        async with semaphore:
            res = await get_prediction_with_usage(facts)
            append_checkpoint(checkpoint_path, event_id, res)
            completed += 1
            if completed % 50 == 0:
                print(f"  прогресс: {completed}/{total}")

    tasks = [worker(row["event_id"], row["facts_text"]) for _, row in todo.iterrows()]
    if tasks:
        await asyncio.gather(*tasks)

    all_done = load_checkpoint(checkpoint_path)
    result = data.copy()
    result["prediction"] = result["event_id"].map(lambda eid: all_done.get(eid, {}).get("prediction"))
    result["input_tokens"] = result["event_id"].map(lambda eid: all_done.get(eid, {}).get("input_tokens"))
    result["output_tokens"] = result["event_id"].map(lambda eid: all_done.get(eid, {}).get("output_tokens"))
    return result


# ═══════════════════════════════════════════════════════════════════════
# ЗАПУСК
# ═══════════════════════════════════════════════════════════════════════

async def main():
    print(f"\nЗапускаю прогон на ПОЛНОМ train ({len(train)} событий)...")
    train_result = await run_full(train, CHECKPOINT_PATH)

    train_result.to_csv("train_with_baseline_predictions.csv", index=False)

    n_errors = train_result["prediction"].isna().sum()
    n_fallback = (train_result["prediction"] == 0.5).sum()
    print(f"\nПропущено/ошибок: {n_errors}")
    print(f"Fallback на 0.5: {n_fallback}/{len(train_result)}")

    # ── R² НА ПОЛНОМ TRAIN — ваш собственный baseline ──
    valid = train_result.dropna(subset=[
        "prediction", "target_percentile_quarter", "surprise_percentile_quarter"
    ])
    X = sm.add_constant(valid[["prediction", "surprise_percentile_quarter"]])
    y = valid["target_percentile_quarter"]
    model = sm.OLS(y, X).fit()
    print(f"\nR² (мой промпт, полный train, n={len(valid)}): {model.rsquared:.4f}")

    # ── СРАВНЕНИЕ С ОФИЦИАЛЬНЫМИ BASELINE НА ТЕХ ЖЕ СОБЫТИЯХ ──
    for col in ["baseline_openai", "baseline_gemini"]:
        if col not in train_result.columns:
            continue
        v = train_result.dropna(subset=[col, "target_percentile_quarter", "surprise_percentile_quarter"])
        if len(v) < 10:
            continue
        Xc = sm.add_constant(v[[col, "surprise_percentile_quarter"]])
        yc = v["target_percentile_quarter"]
        m = sm.OLS(yc, Xc).fit()
        print(f"R² ({col}, n={len(v)}): {m.rsquared:.4f}")

    print("\n✓ Готово. train_with_baseline_predictions.csv сохранён.")


if __name__ == "__main__":
    asyncio.run(main())