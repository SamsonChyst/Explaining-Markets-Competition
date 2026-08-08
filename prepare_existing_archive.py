from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd


ARCHIVE_DIRECTORY = Path("data/archive")
PROCESSED_DIRECTORY = Path("data/processed")

CSV_PATH = PROCESSED_DIRECTORY / "earnings_releases.csv"
PARQUET_PATH = PROCESSED_DIRECTORY / "earnings_releases.parquet"


def extract_ticker(event: dict[str, Any]) -> str | None:
    focal_assets = event.get("focal_assets")

    if not isinstance(focal_assets, list):
        return None

    for asset in focal_assets:
        if not isinstance(asset, dict):
            continue

        if asset.get("identifier_type") == "TICKER":
            ticker = asset.get("identifier_value")

            if isinstance(ticker, str):
                return ticker

    return None


def extract_facts(disclosure: Any) -> list[str]:
    if not isinstance(disclosure, dict):
        return []

    items = disclosure.get("items")

    if not isinstance(items, list):
        return []

    facts: list[str] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        if item.get("kind") != "facts":
            continue

        content = item.get("content")

        if not isinstance(content, list):
            continue

        for fact in content:
            if isinstance(fact, str) and fact.strip():
                facts.append(fact.strip())

    return facts


def extract_car1(
    event: dict[str, Any],
    ticker: str | None,
) -> float | None:
    if ticker is None:
        return None

    event_returns = event.get("event_returns")

    if not isinstance(event_returns, dict):
        return None

    ticker_returns = event_returns.get(ticker)

    if not isinstance(ticker_returns, dict):
        return None

    car1 = ticker_returns.get("car1")

    if isinstance(car1, (int, float)):
        return float(car1)

    return None


def nested_value(data: Any, *keys: str) -> Any:
    current = data

    for key in keys:
        if not isinstance(current, dict):
            return None

        current = current.get(key)

    return current


def extract_baseline_prediction(
    event: dict[str, Any],
    model_name: str,
    ticker: str | None,
) -> float | None:
    if ticker is None:
        return None

    value = nested_value(
        event,
        "baseline_predictions",
        model_name,
        ticker,
    )

    if isinstance(value, (int, float)):
        return float(value)

    return None


def quarter_from_filename(path: Path) -> str:
    # Например:
    # EARNINGS_RELEASE_2026Q2.jsonl.gz
    name = path.name

    if not name.endswith(".jsonl.gz"):
        raise ValueError(f"Неизвестный формат файла: {name}")

    base_name = name.removesuffix(".jsonl.gz")
    quarter = base_name.rsplit("_", maxsplit=1)[-1]

    return quarter


def event_to_row(
    event: dict[str, Any],
    quarter: str,
) -> dict[str, Any]:
    ticker = extract_ticker(event)
    facts = extract_facts(event.get("disclosure"))
    car1 = extract_car1(event, ticker)

    earnings_surprise = nested_value(
        event,
        "metrics",
        "earnings_surprise",
        "surprise",
    )

    if not isinstance(earnings_surprise, (int, float)):
        earnings_surprise = None

    return {
        "event_id": event.get("event_id"),
        "event_type": event.get("event_type"),
        "quarter": quarter,
        "ticker": ticker,
        "timing_category": event.get("timing_category"),
        "event_datetime": event.get("event_datetime"),
        "knowledge_cutoff": event.get("knowledge_cutoff"),
        "status": event.get("status"),
        "return_status": event.get("return_status"),
        "car1": car1,
        "earnings_surprise": earnings_surprise,
        "facts_count": len(facts),
        "facts_json": json.dumps(
            facts,
            ensure_ascii=False,
        ),
        "facts_text": "\n".join(
            f"{number}. {fact}"
            for number, fact in enumerate(facts, start=1)
        ),
        "baseline_openai": extract_baseline_prediction(
            event,
            "openai/ea-explain-contemp-summary",
            ticker,
        ),
        "baseline_gemini": extract_baseline_prediction(
            event,
            "gemini/ea-explain-contemp-summary",
            ticker,
        ),
    }


def read_archive_file(path: Path) -> list[dict[str, Any]]:
    quarter = quarter_from_filename(path)
    rows: list[dict[str, Any]] = []

    with gzip.open(
        path,
        mode="rt",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Ошибка JSON в {path.name}, "
                    f"строка {line_number}: {exc}"
                ) from exc

            if not isinstance(event, dict):
                print(
                    f"[warning] {path.name}, строка {line_number}: "
                    "JSON не является объектом"
                )
                continue

            rows.append(
                event_to_row(
                    event=event,
                    quarter=quarter,
                )
            )

    print(f"[parsed] {path.name}: {len(rows):,} событий")

    return rows


def main() -> None:
    archive_files = sorted(
        ARCHIVE_DIRECTORY.glob("EARNINGS_RELEASE_*.jsonl.gz")
    )

    if not archive_files:
        raise RuntimeError(
            "В data/archive не найдено файлов "
            "EARNINGS_RELEASE_*.jsonl.gz"
        )

    print("Найдены файлы:")

    for path in archive_files:
        print(f"  {path}")

    all_rows: list[dict[str, Any]] = []

    for path in archive_files:
        all_rows.extend(read_archive_file(path))

    df = pd.DataFrame(all_rows)

    if df.empty:
        raise RuntimeError("После обработки датасет пустой")

    df["event_datetime"] = pd.to_datetime(
        df["event_datetime"],
        utc=True,
        errors="coerce",
    )

    df["knowledge_cutoff"] = pd.to_datetime(
        df["knowledge_cutoff"],
        utc=True,
        errors="coerce",
    )

    df["car1"] = pd.to_numeric(
        df["car1"],
        errors="coerce",
    )

    df["earnings_surprise"] = pd.to_numeric(
        df["earnings_surprise"],
        errors="coerce",
    )

    df["target_percentile_quarter"] = (
        df.groupby("quarter")["car1"]
        .rank(
            method="average",
            pct=True,
        )
    )

    df = df.sort_values(
        ["event_datetime", "event_id"],
        na_position="last",
    ).reset_index(drop=True)

    PROCESSED_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        CSV_PATH,
        index=False,
    )

    df.to_parquet(
        PARQUET_PATH,
        index=False,
    )

    print("\nГотово")
    print("Строк:", len(df))
    print("Колонки:", df.columns.tolist())
    print("CSV:", CSV_PATH)
    print("Parquet:", PARQUET_PATH)

    print("\nСобытия по кварталам:")
    print(
        df["quarter"]
        .value_counts()
        .sort_index()
    )

    print("\nПропуски:")
    print(
        df[
            [
                "ticker",
                "car1",
                "earnings_surprise",
                "baseline_openai",
                "baseline_gemini",
                "facts_text",
            ]
        ]
        .isna()
        .sum()
    )


if __name__ == "__main__":
    main()