from pathlib import Path

import pandas as pd


PROCESSED_ROOT = Path("data/processed")
OUTPUT_DIRECTORY = PROCESSED_ROOT / "all"


def main() -> None:
    parquet_files = sorted(
        PROCESSED_ROOT.glob(
            "*/earnings_releases.parquet"
        )
    )

    parquet_files = [
        path
        for path in parquet_files
        if path.parent.name != "all"
    ]

    if not parquet_files:
        raise RuntimeError(
            "Подготовленные квартальные файлы не найдены"
        )

    dataframes = []

    for path in parquet_files:
        df = pd.read_parquet(path)
        dataframes.append(df)

    combined_df = pd.concat(
        dataframes,
        ignore_index=True,
    )

    combined_df = combined_df.sort_values(
        ["event_datetime", "event_id"],
        na_position="last",
    ).reset_index(drop=True)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = OUTPUT_DIRECTORY / "earnings_releases.csv"
    parquet_path = OUTPUT_DIRECTORY / "earnings_releases.parquet"

    combined_df.to_csv(
        csv_path,
        index=False,
    )

    combined_df.to_parquet(
        parquet_path,
        index=False,
    )


if __name__ == "__main__":
    main()