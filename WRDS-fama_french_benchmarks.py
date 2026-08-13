import wrds
import pandas as pd
import numpy as np
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")


def assign_ff12_industry(sic):
    sic = int(sic) if pd.notna(sic) else -1
    if (100 <= sic <= 999) or (2000 <= sic <= 2399) or (2700 <= sic <= 2749) or (2770 <= sic <= 2799) or (
            3100 <= sic <= 3199) or (3940 <= sic <= 3989):
        return "NoDur"
    elif (2500 <= sic <= 2519) or (2590 <= sic <= 2599) or (3630 <= sic <= 3659) or (sic == 3710) or (sic == 3711) or (
            sic == 3714) or (sic == 3716) or (3750 <= sic <= 3751) or (sic == 3792) or (3900 <= sic <= 3939) or (
            3990 <= sic <= 3999):
        return "Durbl"
    elif (2520 <= sic <= 2589) or (2600 <= sic <= 2699) or (2750 <= sic <= 2769) or (3000 <= sic <= 3099) or (
            3200 <= sic <= 3569) or (3580 <= sic <= 3629) or (3700 <= sic <= 3709) or (3712 <= sic <= 3713) or (
            sic == 3715) or (3717 <= sic <= 3749) or (3752 <= sic <= 3791) or (3793 <= sic <= 3799) or (
            3830 <= sic <= 3839) or (3860 <= sic <= 3899):
        return "Manuf"
    elif (1200 <= sic <= 1399) or (2900 <= sic <= 2999):
        return "Enrgy"
    elif (2800 <= sic <= 2829) or (2840 <= sic <= 2899):
        return "Chems"
    elif (3570 <= sic <= 3579) or (3660 <= sic <= 3692) or (3694 <= sic <= 3699) or (3810 <= sic <= 3829) or (
            7370 <= sic <= 7379):
        return "BusEq"
    elif (4800 <= sic <= 4899):
        return "Telcm"
    elif (4900 <= sic <= 4949):
        return "Utils"
    elif (5000 <= sic <= 5999) or (7200 <= sic <= 7299) or (7600 <= sic <= 7699):
        return "Shops"
    elif (2830 <= sic <= 2839) or (sic == 3693) or (3840 <= sic <= 3859) or (8000 <= sic <= 8099):
        return "Hlth"
    elif (6000 <= sic <= 6999):
        return "Money"
    else:
        return "Other"


def safe_divide(numerator, denominator):
    if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
        return np.nan
    return numerator / denominator


def main():
    db = wrds.Connection()

    query = """
        SELECT
            gvkey, datadate, fyearq, fqtr, datacqtr, sic,
            saleq, cogsq, xsgaq, dpq, ebitq, niq, xintq,
            atq, actq, lctq, cheq, dlcq, dlttq, ceqq, invtq, apq,
            prccq, cshoq
        FROM comp.fundq
        WHERE datadate BETWEEN '2024-01-01' AND '2026-03-31'
        AND indfmt = 'INDL'
        AND datafmt = 'STD'
        AND popsrc = 'D'
        AND consol = 'C'
    """

    df = db.raw_sql(query)

    df = df.drop_duplicates(subset=['gvkey', 'datadate'], keep='last')
    df['datadate'] = pd.to_datetime(df['datadate'])
    df = df.sort_values(by=['gvkey', 'datadate'])

    flow_cols = ['saleq', 'cogsq', 'xsgaq', 'dpq', 'ebitq', 'niq', 'xintq']

    for col in flow_cols:
        ltm_col = col.replace('q', '_ltm')
        df[ltm_col] = df.groupby('gvkey')[col].transform(lambda x: x.rolling(window=4, min_periods=4).sum())

    df = df[df['datacqtr'].isin(['2025Q4', '2026Q1'])].copy()

    df = df.dropna(subset=['gvkey', 'sic', 'sale_ltm', 'atq', 'ceqq', 'ni_ltm'])

    df['ff_industry'] = df['sic'].apply(assign_ff12_industry)
    df['ff_industry'] = pd.Categorical(df['ff_industry'],
                                       categories=["NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq", "Telcm",
                                                   "Utils", "Shops", "Hlth", "Money", "Other"])

    df['cash_pct'] = df.apply(lambda row: safe_divide(row['cheq'], row['atq']), axis=1)
    df['current_ratio'] = df.apply(lambda row: safe_divide(row['actq'], row['lctq']), axis=1)
    df['debt_to_equity'] = df.apply(lambda row: safe_divide(row['dlttq'], row['ceqq']), axis=1)

    df['gross_margin_ltm'] = df.apply(lambda row: safe_divide(row['sale_ltm'] - row['cogs_ltm'], row['sale_ltm']),
                                      axis=1)
    df['ebit_margin_ltm'] = df.apply(lambda row: safe_divide(row['ebit_ltm'], row['sale_ltm']), axis=1)
    df['net_margin_ltm'] = df.apply(lambda row: safe_divide(row['ni_ltm'], row['sale_ltm']), axis=1)

    df['roa_ltm'] = df.apply(lambda row: safe_divide(row['ni_ltm'], row['atq']), axis=1)
    df['roe_ltm'] = df.apply(lambda row: safe_divide(row['ni_ltm'], row['ceqq']), axis=1)

    df['dso_ltm'] = df.apply(lambda row: safe_divide(row['actq'], row['sale_ltm']) * 365, axis=1)
    df['dio_ltm'] = df.apply(lambda row: safe_divide(row['invtq'], row['cogs_ltm']) * 365, axis=1)
    df['dpo_ltm'] = df.apply(lambda row: safe_divide(row['apq'], row['cogs_ltm']) * 365, axis=1)

    df['pe_ratio_ltm'] = df.apply(lambda row: safe_divide(row['prccq'], safe_divide(row['ni_ltm'], row['cshoq'])),
                                  axis=1)

    df = df.replace([np.inf, -np.inf], np.nan)
    df.loc[df['ceqq'] <= 0, ['roe_ltm', 'debt_to_equity']] = np.nan
    df.loc[df['sale_ltm'] <= 0, ['gross_margin_ltm', 'ebit_margin_ltm', 'net_margin_ltm']] = np.nan
    df.loc[df['atq'] <= 0, ['roa_ltm']] = np.nan

    metric_cols = [
        'cash_pct', 'current_ratio', 'debt_to_equity',
        'gross_margin_ltm', 'ebit_margin_ltm', 'net_margin_ltm',
        'roa_ltm', 'roe_ltm', 'dso_ltm', 'dio_ltm', 'dpo_ltm', 'pe_ratio_ltm'
    ]

    benchmarks = df.groupby(['ff_industry', 'datacqtr'])[metric_cols].agg([
        ('median', 'median'),
        ('mean', 'mean'),
        ('count', 'count')
    ]).reset_index()

    benchmarks.columns = ['_'.join(col).strip() if col[1] else col[0] for col in benchmarks.columns.values]

    output_file = f"ff12_quarterly_ltm_{datetime.now().strftime('%Y%m%d')}.csv"
    benchmarks.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()