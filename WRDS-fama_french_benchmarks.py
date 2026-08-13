import wrds
import pandas as pd
import numpy as np
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

def assign_ff12_industry(sic):
    sic = int(sic) if pd.notna(sic) else -1
    if (100 <= sic <= 999) or (2000 <= sic <= 2399) or (2700 <= sic <= 2749) or (2770 <= sic <= 2799) or (3100 <= sic <= 3199) or (3940 <= sic <= 3989):
        return "NoDur"
    elif (2500 <= sic <= 2519) or (2590 <= sic <= 2599) or (3630 <= sic <= 3659) or (sic == 3710) or (sic == 3711) or (sic == 3714) or (sic == 3716) or (3750 <= sic <= 3751) or (sic == 3792) or (3900 <= sic <= 3939) or (3990 <= sic <= 3999):
        return "Durbl"
    elif (2520 <= sic <= 2589) or (2600 <= sic <= 2699) or (2750 <= sic <= 2769) or (3000 <= sic <= 3099) or (3200 <= sic <= 3569) or (3580 <= sic <= 3629) or (3700 <= sic <= 3709) or (3712 <= sic <= 3713) or (sic == 3715) or (3717 <= sic <= 3749) or (3752 <= sic <= 3791) or (3793 <= sic <= 3799) or (3830 <= sic <= 3839) or (3860 <= sic <= 3899):
        return "Manuf"
    elif (1200 <= sic <= 1399) or (2900 <= sic <= 2999):
        return "Enrgy"
    elif (2800 <= sic <= 2829) or (2840 <= sic <= 2899):
        return "Chems"
    elif (3570 <= sic <= 3579) or (3660 <= sic <= 3692) or (3694 <= sic <= 3699) or (3810 <= sic <= 3829) or (7370 <= sic <= 7379):
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
            gvkey, fyear, datadate, sic,
            sale, revt, cogs, xsga, dp, ebit, ni, ib, pi, txp, txpd, xint, xsll,
            at, act, lct, che, dlc, dltt, lt, ceq, pstk, pstkl, pstkrv, capx, aqc, lco, ivncf, finv,
            oancf, fcf, div, xrd, dep, am, ap, re, ppent, ppegt, invt,
            prcc_f, csho, epspx, epspi, epsfi
        FROM comp.funda
        WHERE fyear BETWEEN 2015 AND 2026
        AND indfmt = 'INDL'
        AND datafmt = 'STD'
        AND popsrc = 'D'
        AND consol = 'C'
    """
    df = db.raw_sql(query)
    df = df.drop_duplicates(subset=['gvkey', 'fyear', 'datadate'], keep='first')
    df = df.dropna(subset=['gvkey', 'fyear', 'sic', 'sale', 'at', 'ceq', 'ni'])
    df['ff_industry'] = df['sic'].apply(assign_ff12_industry)
    df['ff_industry'] = pd.Categorical(df['ff_industry'],
        categories=["NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq", "Telcm", "Utils", "Shops", "Hlth", "Money", "Other"])

    # Common-size balance sheet
    df['cash_pct'] = safe_divide(df['che'], df['at'])
    df['ar_pct'] = safe_divide(df['act'], df['at'])
    df['inventory_pct'] = safe_divide(df['ivncf'], df['at'])
    df['ppe_pct'] = safe_divide(df['ppent'], df['at'])
    df['current_liab_pct'] = safe_divide(df['lct'], df['at'])
    df['lt_debt_pct'] = safe_divide(df['dltt'], df['at'])
    df['total_debt_pct'] = safe_divide(df['dlc'] + df['dltt'], df['at'])
    df['equity_pct'] = safe_divide(df['ceq'], df['at'])
    df['working_capital_pct'] = safe_divide(df['act'] - df['lct'], df['at'])

    # Common-size income statement
    df['cogs_pct'] = safe_divide(df['cogs'], df['sale'])
    df['sgna_pct'] = safe_divide(df['xsga'], df['sale'])
    df['dep_amort_pct'] = safe_divide(df['dep'] + df['am'], df['sale'])
    df['ebit_pct'] = safe_divide(df['ebit'], df['sale'])
    df['interest_pct'] = safe_divide(df['xint'], df['sale'])
    df['pretax_pct'] = safe_divide(df['pi'], df['sale'])
    df['tax_pct'] = safe_divide(df['txp'], df['pi'])
    df['net_income_pct'] = safe_divide(df['ni'], df['sale'])
    df['rd_pct'] = safe_divide(df['xrd'], df['sale'])

    # Profitability
    df['roa'] = safe_divide(df['ni'], df['at'])
    df['roe'] = safe_divide(df['ni'], df['ceq'])
    df['gross_margin'] = safe_divide(df['sale'] - df['cogs'], df['sale'])
    df['ebit_margin'] = df['ebit_pct']
    df['net_margin'] = df['net_income_pct']
    df['roic'] = safe_divide(df['ebit'] * (1 - df['txp'] / df['pi']), df['at'] - df['lct'])

    # Liquidity
    df['current_ratio'] = safe_divide(df['act'], df['lct'])
    df['quick_ratio'] = safe_divide(df['act'] - df['ivncf'], df['lct'])
    df['cash_ratio'] = safe_divide(df['che'], df['lct'])

    # Solvency
    df['debt_to_equity'] = safe_divide(df['dltt'], df['ceq'])
    df['debt_to_assets'] = safe_divide(df['dlc'] + df['dltt'], df['at'])
    df['equity_to_assets'] = safe_divide(df['ceq'], df['at'])
    df['interest_coverage'] = safe_divide(df['ebit'], df['xint'])
    df['fixed_charge_coverage'] = safe_divide(df['ebit'] + df['xint'], df['xint'] + df['dp'])

    # Efficiency
    df['asset_turnover'] = safe_divide(df['sale'], df['at'])
    df['inventory_turnover'] = safe_divide(df['cogs'], df['ivncf'])
    df['receivables_turnover'] = safe_divide(df['sale'], df['act'])
    df['payables_turnover'] = safe_divide(df['cogs'], df['ap'])
    df['dso'] = safe_divide(df['act'], df['sale']) * 365
    df['dio'] = safe_divide(df['ivncf'], df['cogs']) * 365
    df['dpo'] = safe_divide(df['ap'], df['cogs']) * 365

    # Cash flow
    df['ocf_to_revenue'] = safe_divide(df['oancf'], df['sale'])
    df['fcf_to_revenue'] = safe_divide(df['fcf'], df['sale'])
    df['fcf_yield'] = safe_divide(df['fcf'], df['at'])
    df['cash_conversion'] = safe_divide(df['oancf'], df['ni'])
    df['fcf_conversion'] = safe_divide(df['fcf'], df['ni'])
    df['capex_to_sales'] = safe_divide(df['capx'], df['sale'])
    df['capex_to_assets'] = safe_divide(df['capx'], df['at'])

    # Growth rates
    df = df.sort_values(['gvkey', 'fyear'])
    df['revenue_growth'] = df.groupby('gvkey')['sale'].pct_change()
    df['ni_growth'] = df.groupby('gvkey')['ni'].pct_change()
    df['ebit_growth'] = df.groupby('gvkey')['ebit'].pct_change()
    df['ocf_growth'] = df.groupby('gvkey')['oancf'].pct_change()
    df['capex_growth'] = df.groupby('gvkey')['capx'].pct_change()

    # Valuation
    df['pe_ratio'] = safe_divide(df['prcc_f'], df['epspx'])
    df['pb_ratio'] = safe_divide(df['prcc_f'], df['ceq'] / df['csho'])
    df['ev_to_ebitda'] = safe_divide(df['at'] + df['dlc'] + df['dltt'] - df['che'],
                                     df['ebit'] + df['dep'] + df['am'])

    # YoY changes for all ratios
    ratio_cols = ['roa', 'roe', 'gross_margin', 'ebit_margin', 'net_margin', 'current_ratio',
                  'quick_ratio', 'debt_to_equity', 'debt_to_assets', 'asset_turnover',
                  'inventory_turnover', 'receivables_turnover', 'interest_coverage']
    for col in ratio_cols:
        df[f'{col}_yoy'] = df.groupby('gvkey')[col].pct_change()

    # Handle edge cases
    df = df.replace([np.inf, -np.inf], np.nan)
    df.loc[df['ceq'] <= 0, ['roe', 'debt_to_equity']] = np.nan
    df.loc[df['sale'] <= 0, ['gross_margin', 'ebit_margin', 'net_margin']] = np.nan
    df.loc[df['at'] <= 0, ['roa', 'roic']] = np.nan

    # Aggregate by industry-year
    metric_cols = [
        'cash_pct', 'ar_pct', 'inventory_pct', 'ppe_pct', 'current_liab_pct',
        'lt_debt_pct', 'total_debt_pct', 'equity_pct', 'working_capital_pct',
        'cogs_pct', 'sgna_pct', 'dep_amort_pct', 'ebit_pct', 'interest_pct',
        'pretax_pct', 'tax_pct', 'net_income_pct', 'rd_pct',
        'roa', 'roe', 'roic', 'gross_margin', 'ebit_margin', 'net_margin',
        'current_ratio', 'quick_ratio', 'cash_ratio',
        'debt_to_equity', 'debt_to_assets', 'equity_to_assets', 'interest_coverage',
        'fixed_charge_coverage', 'asset_turnover', 'inventory_turnover',
        'receivables_turnover', 'payables_turnover', 'dso', 'dio', 'dpo',
        'ocf_to_revenue', 'fcf_to_revenue', 'fcf_yield', 'cash_conversion',
        'fcf_conversion', 'capex_to_sales', 'capex_to_assets',
        'revenue_growth', 'ni_growth', 'ebit_growth', 'ocf_growth', 'capex_growth',
        'pe_ratio', 'pb_ratio', 'ev_to_ebitda'
    ] + [f'{col}_yoy' for col in ratio_cols]

    benchmarks = df.groupby(['ff_industry', 'fyear']).agg(
        median=('median', 'median'),
        q1=('q1', lambda x: x.quantile(0.25)),
        q3=('q3', lambda x: x.quantile(0.75)),
        mean=('mean', 'mean'),
        std=('std', 'std'),
        count=('count', 'count')
    ).reset_index()

    benchmarks.columns = ['_'.join(col).strip() for col in benchmarks.columns.values]
    benchmarks = benchmarks.rename(columns={'ff_industry_': 'ff_industry', 'fyear_': 'fyear'})

    base_rates = {
        "NoDur": 0.50, "Durbl": 0.55, "Manuf": 0.55, "Enrgy": 0.60,
        "Chems": 0.55, "BusEq": 0.65, "Telcm": 0.50, "Utils": 0.45,
        "Shops": 0.50, "Hlth": 0.60, "Money": 0.40, "Other": 0.50
    }
    benchmarks['base_rate'] = benchmarks['ff_industry'].map(base_rates)

    output_file = f"ff12_benchmarks_{datetime.now().strftime('%Y%m%d')}.csv"
    benchmarks.to_csv(output_file, index=False)

    print(f"Saved {len(benchmarks)} industry-year benchmarks to {output_file}")

if __name__ == "__main__":
    main()