import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import wrds

warnings.filterwarnings("ignore")

# Fama-French 12 industry classification.
FF12_CATEGORIES = [
    "NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq",
    "Telcm", "Utils", "Shops", "Hlth", "Money", "Other",
]

def assign_ff12_industry(sic):
    """Assign a SIC code to the Fama-French 12-industry classification."""
    try:
        sic = int(float(sic)) if pd.notna(sic) else -1
    except (ValueError, TypeError):
        return "Other"

    if (
        100 <= sic <= 999
        or 2000 <= sic <= 2399
        or 2700 <= sic <= 2749
        or 2770 <= sic <= 2799
        or 3100 <= sic <= 3199
        or 3940 <= sic <= 3989
    ):
        return "NoDur"

    if (
        2500 <= sic <= 2519
        or 2590 <= sic <= 2599
        or 3630 <= sic <= 3659
        or sic in (3710, 3711, 3714, 3716)
        or 3750 <= sic <= 3751
        or sic == 3792
        or 3900 <= sic <= 3939
        or 3990 <= sic <= 3999
    ):
        return "Durbl"

    if (
        2520 <= sic <= 2589
        or 2600 <= sic <= 2699
        or 2750 <= sic <= 2769
        or 3000 <= sic <= 3099
        or 3200 <= sic <= 3569
        or 3580 <= sic <= 3629
        or 3700 <= sic <= 3709
        or 3712 <= sic <= 3713
        or sic == 3715
        or 3717 <= sic <= 3749
        or 3752 <= sic <= 3791
        or 3793 <= sic <= 3799
        or 3830 <= sic <= 3839
        or 3860 <= sic <= 3899
    ):
        return "Manuf"

    if 1200 <= sic <= 1399 or 2900 <= sic <= 2999:
        return "Enrgy"

    if 2800 <= sic <= 2829 or 2840 <= sic <= 2899:
        return "Chems"

    if (
        3570 <= sic <= 3579
        or 3660 <= sic <= 3692
        or 3694 <= sic <= 3699
        or 3810 <= sic <= 3829
        or 7370 <= sic <= 7379
    ):
        return "BusEq"

    if 4800 <= sic <= 4899:
        return "Telcm"

    if 4900 <= sic <= 4949:
        return "Utils"

    if (
        5000 <= sic <= 5999
        or 7200 <= sic <= 7299
        or 7600 <= sic <= 7699
    ):
        return "Shops"

    if (
        2830 <= sic <= 2839
        or sic == 3693
        or 3840 <= sic <= 3859
        or 8000 <= sic <= 8099
    ):
        return "Hlth"

    if 6000 <= sic <= 6999:
        return "Money"

    return "Other"

def safe_divide(numerator, denominator):
    """Element-wise-safe scalar division helper."""
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator

def add_consecutive_quarter_ltm(df, flow_cols):
    """
    Calculate LTM values only when the four observations are consecutive
    fiscal quarters.
    """
    df = df.copy()
    quarter_index = (
        pd.to_numeric(df["fyearq"], errors="coerce") * 4
        + pd.to_numeric(df["fqtr"], errors="coerce")
    )
    df["_quarter_index"] = quarter_index

    for col in flow_cols:
        ltm_col = col.replace("q", "_ltm")
        df[ltm_col] = (
            df.groupby("gvkey", sort=False)[col]
            .transform(lambda x: x.rolling(4, min_periods=4).sum())
        )

    quarter_gap = (
        df.groupby("gvkey", sort=False)["_quarter_index"]
        .transform(lambda x: x - x.shift(3))
    )
    invalid_ltm = quarter_gap != 3
    ltm_cols = [col.replace("q", "_ltm") for col in flow_cols]
    df.loc[invalid_ltm, ltm_cols] = np.nan

    return df.drop(columns="_quarter_index")

def build_benchmarks(df):
    """Build FF12 x quarter cross-sectional benchmark statistics."""
    metric_cols = [
        "cash_pct", "current_ratio", "long_term_debt_to_equity",
        "gross_margin_ltm", "ebit_margin_ltm", "net_margin_ltm",
        "roa_ltm", "roe_ltm", "dso_ltm", "dio_ltm", "dpo_ltm", "pe_ratio_ltm",
    ]

    benchmarks = (
        df.groupby(
            ["ff_industry", "datacqtr"],
            observed=False,  # Force all 12 industries to appear
        )[metric_cols]
        .agg([
            ("median", "median"),
            ("p25", lambda x: x.quantile(0.25)),
            ("p75", lambda x: x.quantile(0.75)),
            ("mean", "mean"),
            ("count", "count"),
        ])
        .reset_index()
    )

    # Flatten MultiIndex columns
    flattened = []
    for col in benchmarks.columns:
        if isinstance(col, tuple):
            flattened.append(f"{col[0]}_{col[1]}" if col[1] else col[0])
        else:
            flattened.append(col)
    benchmarks.columns = flattened

    return benchmarks

def main():
    db = wrds.Connection()

    # --- Load data ---
    query = """
        SELECT
            gvkey, datadate, fyearq, fqtr, datacqtr, sic,
            saleq, cogsq, xsgaq, dpq, ebitq, niq, xintq,
            atq, actq, lctq, cheq, dlcq, dlttq, ceqq,
            invtq, apq, rectq, prccq, cshoq
        FROM comp.fundq
        WHERE datadate BETWEEN '2024-01-01' AND '2026-03-31'
          AND indfmt = 'INDL'
          AND datafmt = 'STD'
          AND popsrc = 'D'
          AND consol = 'C'
        ORDER BY gvkey, datadate
    """
    print("Downloading Compustat quarterly data...")
    df = db.raw_sql(query)

    if df.empty:
        raise RuntimeError("WRDS returned no observations.")

    # --- Clean data ---
    df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")

    # Clean SIC: handle NaN, non-numeric, and 6-digit codes
    df["sic"] = pd.to_numeric(df["sic"], errors="coerce")
    df["sic"] = df["sic"].fillna(-1)  # Missing SIC -> -1 (maps to "Other")
    # Extract first 4 digits if SIC is stored as 6-digit (e.g., 123456 -> 1234)
    df["sic"] = df["sic"].astype(str).str[:4].astype(float)

    # --- Assign FF12 industry FIRST (before any filtering) ---
    df["ff_industry"] = df["sic"].apply(assign_ff12_industry)
    df["ff_industry"] = pd.Categorical(
        df["ff_industry"],
        categories=FF12_CATEGORIES,
    )
    print("FF12 Industry counts (after assignment):")
    print(df["ff_industry"].value_counts(dropna=False))

    # --- Convert numeric columns ---
    numeric_cols = [
        "fyearq", "fqtr", "sic", "saleq", "cogsq", "xsgaq", "dpq",
        "ebitq", "niq", "xintq", "atq", "actq", "lctq", "cheq",
        "dlcq", "dlttq", "ceqq", "invtq", "apq", "rectq", "prccq", "cshoq",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Deduplicate ---
    df = df.drop_duplicates(
        subset=["gvkey", "datadate"],
        keep="last",
    ).sort_values(["gvkey", "fyearq", "fqtr", "datadate"]).reset_index(drop=True)

    # --- LTM calculations ---
    flow_cols = ["saleq", "cogsq", "xsgaq", "dpq", "ebitq", "niq", "xintq"]
    df = add_consecutive_quarter_ltm(df, flow_cols)

    # --- Filter for target quarters ---
    df = df[df["datacqtr"].isin(["2025Q4", "2026Q1"])].copy()

    # --- Drop rows with missing critical values ---
    df = df.dropna(
        subset=["gvkey", "sic", "sale_ltm", "atq", "ceqq", "ni_ltm"]
    )
    print("FF12 Industry counts (after filtering):")
    print(df["ff_industry"].value_counts(dropna=False))

    # --- Calculate ratios ---
    df["cash_pct"] = df.apply(lambda row: safe_divide(row["cheq"], row["atq"]), axis=1)
    df["current_ratio"] = df.apply(lambda row: safe_divide(row["actq"], row["lctq"]), axis=1)
    df["long_term_debt_to_equity"] = df.apply(
        lambda row: safe_divide(row["dlttq"], row["ceqq"]), axis=1
    )
    df["gross_margin_ltm"] = df.apply(
        lambda row: safe_divide(row["sale_ltm"] - row["cogs_ltm"], row["sale_ltm"]), axis=1
    )
    df["ebit_margin_ltm"] = df.apply(
        lambda row: safe_divide(row["ebit_ltm"], row["sale_ltm"]), axis=1
    )
    df["net_margin_ltm"] = df.apply(
        lambda row: safe_divide(row["ni_ltm"], row["sale_ltm"]), axis=1
    )
    df["roa_ltm"] = df.apply(lambda row: safe_divide(row["ni_ltm"], row["atq"]), axis=1)
    df["roe_ltm"] = df.apply(lambda row: safe_divide(row["ni_ltm"], row["ceqq"]), axis=1)
    df["dso_ltm"] = df.apply(
        lambda row: safe_divide(row["rectq"], row["sale_ltm"]) * 365, axis=1
    )
    df["dio_ltm"] = df.apply(
        lambda row: safe_divide(row["invtq"], row["cogs_ltm"]) * 365, axis=1
    )
    df["dpo_ltm"] = df.apply(
        lambda row: safe_divide(row["apq"], row["cogs_ltm"]) * 365, axis=1
    )
    df["eps_ltm"] = df.apply(lambda row: safe_divide(row["ni_ltm"], row["cshoq"]), axis=1)
    df["pe_ratio_ltm"] = df.apply(
        lambda row: safe_divide(row["prccq"], row["eps_ltm"]), axis=1
    )
    df.loc[df["ni_ltm"] <= 0, "pe_ratio_ltm"] = np.nan

    # --- Sanity filtering ---
    df = df.replace([np.inf, -np.inf], np.nan)
    df.loc[df["ceqq"] <= 0, ["roe_ltm", "long_term_debt_to_equity"]] = np.nan
    df.loc[df["sale_ltm"] <= 0, ["gross_margin_ltm", "ebit_margin_ltm", "net_margin_ltm", "dso_ltm"]] = np.nan
    df.loc[df["atq"] <= 0, "roa_ltm"] = np.nan
    df.loc[df["cogs_ltm"] <= 0, ["dio_ltm", "dpo_ltm"]] = np.nan

    # --- Build benchmarks ---
    benchmarks = build_benchmarks(df)

    # --- Save outputs ---
    timestamp = datetime.now().strftime("%Y%m%d")
    benchmarks_file = f"ff12_quarterly_ltm_{timestamp}.csv"
    company_file = f"ff12_company_ratios_{timestamp}.csv"

    benchmarks.to_csv(benchmarks_file, index=False)

    company_cols = [
        "gvkey", "datadate", "datacqtr", "fyearq", "fqtr", "sic", "ff_industry",
        "cash_pct", "current_ratio", "long_term_debt_to_equity",
        "gross_margin_ltm", "ebit_margin_ltm", "net_margin_ltm",
        "roa_ltm", "roe_ltm", "dso_ltm", "dio_ltm", "dpo_ltm",
        "eps_ltm", "pe_ratio_ltm",
    ]
    df[company_cols].to_csv(company_file, index=False)

    # --- Summary ---
    print("\n=== Summary ===")
    print(f"Company observations: {len(df):,}")
    print(f"Industries present: {sorted(df['ff_industry'].unique())}")
    print(f"Quarters: {sorted(df['datacqtr'].dropna().unique())}")
    print(f"Benchmark rows: {len(benchmarks):,}")
    print(f"Benchmarks saved to: {benchmarks_file}")
    print(f"Company ratios saved to: {company_file}")

if __name__ == "__main__":
    main()