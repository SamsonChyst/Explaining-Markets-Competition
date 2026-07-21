"""
This module is used to download and process Transcripts and further export
them into .csv format
"""
import json
import requests
import time
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
ROIC_AI_API = os.getenv('ROIC_AI_API')


def download_transcripts(ticker: str):
    ticker = ticker.upper()
    output_dir = f'transcripts_{ticker}'

    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        return

    headers = {'User-Agent': 'Mozilla/5.0'}

    # list of earning calls
    list_url = f'https://api.roic.ai/v2/company/earnings-calls/list/{ticker}?apikey={ROIC_AI_API}'

    try:
        response = requests.get(list_url, headers=headers, timeout=10)
        time.sleep(12)

        if response.status_code != 200:
            return

        calls_list = response.json()
    except (requests.exceptions.RequestException, json.JSONDecodeError):
        return

    if not isinstance(calls_list, list):
        return

    os.makedirs(output_dir, exist_ok=True)

    # listing the years/quarters of calls
    min_year = 2024
    filtered_calls = [
        call for call in calls_list if call.get('year', 0) >= min_year
    ]

    # downloading calls into transcripts_{ticker}
    for call in filtered_calls:
        year = call.get('year')
        quarter = call.get('quarter')

        if not year or not quarter:
            continue

        file_name = f'{output_dir}/{ticker}_{year}_Q{quarter}.json'

        if os.path.exists(file_name):
            # skip
            continue

        print(f'Downloading transcript for {ticker} {year} Q{quarter}')

        # API
        transcript_url = f'https://api.roic.ai/v2/company/earnings-calls/transcript/{ticker}/{year}/{quarter}?apikey={ROIC_AI_API}'

        try:
            resp = requests.get(transcript_url, headers=headers, timeout=10)

            if resp.status_code == 200:
                transcript_data = resp.json()
                # writing jsons

                with open(file_name, 'w', encoding='utf-8') as f:
                    json.dump(transcript_data, f, ensure_ascii=False, indent=4)
                print(f'{file_name} Downloaded')

            else:
                print(
                    f'Error {year} Q{quarter}: Status - {resp.status_code}'
                )
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            print(f'Error {year} Q{quarter}: Status - {e}')

        # 5 responses in a minute - max
        time.sleep(12)

if __name__ == '__main__':
    df = pd.read_csv('companies.csv')[['ticker', 'industry']]
    tickers = df['ticker'].dropna().astype(str).str.strip()

    for ticker in tickers:
        download_transcripts(ticker)