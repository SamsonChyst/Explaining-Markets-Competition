"""
This module is used to download and process Transcripts and further export
them into .csv format

Transcripts are downloaded from ROIC.AI at 5 per minute
Their Summary is made via Mistral Large Latest
"""
import json
import os
import time
import pandas as pd
import requests
from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()
ROIC_AI_API = os.getenv('ROIC_AI_API')
MISTRAL_KEY = os.getenv('MISTRAL_KEY')
client = Mistral(api_key=MISTRAL_KEY)

MAX_RETRIES = 5
BASE_SLEEP = 12
MISTRAL_SUMMARY_PROMPT = '''
You are a financial analyst distilling earnings-call transcripts into dense, fact-first summaries for a training dataset.

OUTPUT
Return only the summary itself: plain-text sentences, one per line, no markdown, no numbering or bullets, no quotation marks, no preamble, no closing remark. Write in English regardless of the transcript's source language.

LENGTH
8-12 sentences. Let the transcript's density of decision-relevant facts set the count naturally — don't pad to reach a number or force one down.

STYLE CALIBRATION (tone and density reference only — unrelated company, ignore the topic, match the register)
Q2 2026 revenue rose 5% year-over-year to $10.9 billion with growth across all four segments, and sequential sales increased 10%.
The company delivered $20 billion in net awards for a 1.84x quarterly book-to-bill ratio and grew backlog 17% year-over-year to a record $105 billion.
Q2 EPS of $7.68 benefited significantly from a lower effective tax rate, including a remeasurement of uncertain tax positions and a gain on the sale of an equity investment.
Management raised full-year 2026 guidance, lifting sales to ~$44 billion (over 5% organic growth), raising EPS by $1.20 to $28.60-$29.10, and increasing full-year book-to-bill to at least 1.25x.
Space margins were pressured by an unfavorable GEM 63XL EAC adjustment tied to a Q1 launch anomaly, prompting the company to lower full-year Space margin guidance to the low 10% range with redesigned motors expected to begin delivering by year-end.
2026 CapEx is expected at $1.85 billion, rising to roughly 4.5% of sales in 2027 and 2028 to support the B-21 ramp, while Q2 adjusted free cash flow reached nearly $1 billion and full-year free cash flow guidance of $3.1-$3.5 billion was reaffirmed.

SOURCING
Draw primarily from the CEO's and CFO's prepared remarks, before the operator opens the floor to analyst questions — that's where management states its own priorities and where the real numbers concentrate. Pull a fact from the Q&A only when it discloses a number that never appeared in the prepared remarks (for example, a segment figure given only in an analyst answer); skip Q&A exchanges that just restate or elaborate on something already covered.

WHAT TO COVER
Write one sentence for each item below that the transcript addresses with a number; skip anything it doesn't mention rather than inventing a placeholder.
- Headline revenue or organic growth: period, year-over-year change, primary driver, sequential change if stated.
- The most-emphasized non-revenue metric of the quarter (bookings, backlog, book-to-bill, orders, volume/price mix).
- EPS or the headline profit metric, plus any one-time item that moved it (tax rate, gain or loss, comp reset).
- Updated guidance for this period and/or the next, stated against the prior guide.
- Any other explicit forward guidance (a segment, margin, cash flow, capex).
- Two to four segment or product-line facts — prioritize the largest swings and any one-time or unusual item (impairment, FX, divestiture, program-specific charge).
- Margin trend (gross or operating) and its stated driver.
- Cash flow, capital return (buybacks/dividends), or leverage/net debt.

Three underlying sentence shapes to draw on naturally — don't force the exact wording, just the structure:
1. Historical performance: "{metric} in {period} reached {value}, a {change} change year-over-year, driven by {driver}."
2. Forward guidance: "Management projects {metric} to {trend} to {target} in {period}, driven by {catalyst}."
3. Balance sheet / asset quality: "{metric} stood at {value} ({ratio} of total assets/backlog), reflecting {driver}."

RULES
- Every sentence needs at least one number copied directly from the transcript. Never estimate, round dramatically, or calculate a derived figure yourself from two raw numbers — use only rates or percentages the speaker states explicitly.
- No names of people — refer to "management," "the company," or the segment/program name.
- No hedging verbs ("said," "noted," "mentioned," "believes") and no unquantified filler adjectives ("strong," "significant," "robust") — state the fact directly, or pair the adjective with its number.
- One sentence covers one topic. You may join two tightly related figures with "and" or "while" (a headline number plus its sequential change, for instance), but never combine two unrelated facts.

Before finalizing, confirm every number traces back to something explicit in the transcript and that no person's name appears anywhere in the output.
'''


def download_transcripts(ticker: str):
    ticker = ticker.upper()
    output_dir = f'Transcripts/{ticker}'

    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        return

    headers = {'User-Agent': 'Mozilla/5.0'}

    # list of earning calls
    list_url = f'https://api.roic.ai/v2/company/earnings-calls/list/{ticker}?apikey={ROIC_AI_API}'

    try:
        response = requests.get(list_url, headers=headers, timeout=10)
        # 5 responses in a minute - max
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
        transcript_url = f'https://api.roic.ai/v2/company/earnings-calls/transcript/{ticker}'
        params = {
            'apikey': ROIC_AI_API,
            'year': year,
            'quarter': quarter
        }

        try:
            resp = requests.get(transcript_url, headers=headers, params=params, timeout=10)

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


def summary_extract(ticker: str):
    """
    Mutates jsons to summarize transcripts in Transcripts/{ticker}/{json}
    """
    ticker = ticker.upper()
    input_dir = f'Transcripts/{ticker}'

    print(f'Start: {ticker}')

    if not os.path.isdir(input_dir):
        print(f'Folder not found: {input_dir}')
        return

    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith('.json'):
            continue

        print(f'Processing: {filename}')

        file_path = os.path.join(input_dir, filename)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f'Error reading {filename}: {e}')
            continue

        if transcript_data.get('summary'):
            print(f'Summary exists: {filename}')
            continue

        transcript = transcript_data.get('content')

        if not transcript:
            print(f'No content: {filename}')
            continue

        success = False

        for attempt in range(MAX_RETRIES):
            try:
                print(f'Sending: {filename}')

                response = client.chat.complete(
                    model='mistral-large-latest',
                    messages=[
                        {
                            'role': 'user',
                            'content': f'{MISTRAL_SUMMARY_PROMPT}, here is the transcript to be summarized: {transcript}'
                        }
                    ],
                )

                print(f'Received: {filename}')

                transcript_data['summary'] = response.choices[0].message.content
                success = True
                break

            except Exception as e:
                error_msg = str(e).lower()

                if '429' in error_msg or 'rate limit' in error_msg or 'too many requests' in error_msg:
                    time.sleep(5 * (2 ** attempt))
                else:
                    print(f'Error: {filename}: {e}')
                    break

        if not success:
            continue

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(transcript_data, f, ensure_ascii=False, indent=4)

            print(f'Saved: {filename}')
            time.sleep(BASE_SLEEP)

        except OSError as e:
            print(f'Write error: {filename}: {e}')


if __name__ == '__main__':
    df = pd.read_csv('companies.csv')[['ticker', 'industry']]
    tickers = df['ticker'].dropna().astype(str).str.strip()

    # ROIC AI parse
    '''
    for ticker in tickers:
        ticker = ticker.upper()
        output_dir = f'Transcripts/{ticker}'

        if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
            print(f'Skipping {ticker}: directory already exists and is not empty')
            continue

        download_transcripts(ticker)
    '''

    # Mistral AI summary correction
    for ticker in os.listdir('Transcripts/'):
        summary_extract(ticker)