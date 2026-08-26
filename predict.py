"""
`predict(event)` is called once per competition event, after the webhook has
already been verified for you. Return one prediction per focal asset. Everything
else in this repo (webhook verification, dedupe, submission) is plumbing.

This implementation asks a Mistral model for a calibrated percentile via chat
completion, per focal asset. For each asset, THREE calls are made:
  1. `_classify_ff12`      — determine the FF12 industry from the facts.
  2. `_predict_percentile` — predict the percentile, given that industry's
                              (Gross Margin %, ROE %) benchmark handed to it
                              directly, instead of asking the model to pick
                              one out of all twelve benchmarks in-context.
  3. `_sanity_check`       — a second pass shown the SAME facts plus the
                              percentile from step 2, asked whether that
                              percentile looks grounded in the facts or
                              hallucinated. A "FLAG" verdict overrides the
                              prediction with the 0.5 placeholder rather than
                              submitting something that failed its own audit.

FF12 industry benchmarks (Gross Margin %, ROE %) are aggregated from Wharton
Research Data Services (WRDS) — see FF12_BENCHMARKS below.

`MISTRAL_KEY` is required — the module raises immediately at import time if
it's missing, so a missing key is caught at deploy time rather than mid-run.
A single failed call to the Mistral API (network/API error, or a reply that
can't be parsed) is retried a few times before falling back to a safe default
(industry "Other" for classification, percentile 0.5 for prediction, "OK"/
pass-through for a sanity check that itself couldn't be completed), so one
bad event doesn't stall the whole prediction run.

RATE LIMITING — READ THIS IF YOU SEE 429s:
Each Mistral key has its own account-wide request budget. On this account
it's observed (via the `x-ratelimit-limit-req-minute` response header) to be
4 requests/minute — NOT 3, and NOT a token-based limit; the 429 bodies show
`"type":"rate_limited"` tied specifically to request count. That budget is
shared across *every* call made with that key, from *every event* —
including events whose `predict()` calls overlap in time, which happens
routinely once more than one earnings event lands in the same short window
(this is exactly what happened: JKHY's and EL's calls interleaved within the
same calendar minute and jointly exhausted the account's 4-request budget,
even though each event's own 20s-spaced call sequence never would have on
its own).

A per-event sleep between that event's own calls cannot see or coordinate
with another event's calls running concurrently on the same key. The fix is
`_RateLimiter`: a single lock-protected sliding window, shared by every
caller of a given key — across threads AND across separate `predict()`
invocations within this process — that every Mistral call must acquire a
slot from (including retries) before it's allowed to fire. This replaces the
old fixed-interval `time.sleep(REQUEST_SLEEP_SECONDS)` pacing, which only
ever protected a single event's own call sequence, not the account's actual
shared budget.

Caveat: this only coordinates calls made *within this process*. If
MISTRAL_KEY_1 or MISTRAL_KEY_2 is ALSO used by some other script, process, or
teammate outside this module, this limiter can't see that traffic and you
can still occasionally hit 429 from it — lower MISTRAL_MAX_REQUESTS_PER_MINUTE
below the account's real limit to leave headroom for that case.

DUAL-KEY PATH: if an event has >= DUAL_KEY_THRESHOLD focal assets AND
MISTRAL_KEY_2 is configured, the focal assets are split roughly in half and
run CONCURRENTLY across two threads, each using its own Mistral key end to
end (classify + predict + sanity, all three calls, for its own half). This
is meant for two people's own independent Mistral accounts contributing to
one shared job — see the .env notes shipped alongside this file for what
this is (and is not) safe to do under Mistral's terms. If MISTRAL_KEY_2
isn't set, every event just runs the single-key sequential path, same as
before — this file is safe to deploy either way. Each key gets its own
`_RateLimiter` instance, since the two accounts have independent budgets.
"""

from __future__ import annotations
import json
import os
import re
import threading
import time
from collections import deque
import httpx
from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()
MISTRAL_KEY_1 = os.getenv("MISTRAL_KEY_1")

if not MISTRAL_KEY_1:
    raise EnvironmentError(
        "MISTRAL_KEY is not set in the environment."
    )

# Optional second account's key. When set (and the event is large enough,
# see DUAL_KEY_THRESHOLD), that account's own independent rate-limit budget
# is used to run a second half of the focal assets concurrently with the
# first. Must be a key from a DIFFERENT Mistral account than MISTRAL_KEY,
# controlled by whoever owns it — never your own second account, and never
# a key someone else handed you. See the .env notes for why.
MISTRAL_KEY_2 = os.getenv("MISTRAL_KEY_2")

client = Mistral(api_key=MISTRAL_KEY_1)
client_2 = Mistral(api_key=MISTRAL_KEY_2) if MISTRAL_KEY_2 else None

MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-latest")

# One retryable call retries this many times (including the first attempt)
# before giving up and returning its fallback.
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0

# --- Rate limiting -------------------------------------------------------
#
# Each key's account-wide request budget, shared by EVERY call made with
# that key across every thread and every event (this is what actually
# protects the account, unlike a per-event sleep — see module docstring).
# Set this from the `x-ratelimit-limit-req-minute` header on your own
# account's responses (visible in a 429 error, or on Mistral's console
# Limits page) — 4 is what this account showed; lower it a bit if you know
# something outside this process also shares the same key.
MISTRAL_MAX_REQUESTS_PER_MINUTE = int(os.getenv("MISTRAL_MAX_REQUESTS_PER_MINUTE", "4"))

# Independent budget for the second account, in case it's on a different
# tier. Defaults to the same value as the first.
MISTRAL_MAX_REQUESTS_PER_MINUTE_2 = int(
    os.getenv("MISTRAL_MAX_REQUESTS_PER_MINUTE_2", str(MISTRAL_MAX_REQUESTS_PER_MINUTE))
)

# Small minimum gap enforced between any two consecutive requests on the
# same key, even when the per-minute window still has room — a cheap guard
# against also tripping a separate per-second cap that a per-minute-only
# limiter wouldn't otherwise account for.
MISTRAL_MIN_REQUEST_GAP_SECONDS = float(os.getenv("MISTRAL_MIN_REQUEST_GAP_SECONDS", "1.0"))


class _RateLimiter:
    """Thread-safe sliding-window limiter shared by every caller of one key.

    `acquire()` blocks until a slot is free, then reserves it immediately,
    so it's safe to call from multiple threads (the dual-key path) AND from
    multiple overlapping `predict()` invocations (multiple events processing
    close together in time) — both share the SAME instance per key, which is
    what a per-event `time.sleep()` never could.
    """

    def __init__(self, max_requests: int, window_seconds: float = 60.0, min_gap_seconds: float = 1.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.min_gap_seconds = min_gap_seconds
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= self.window_seconds:
                    self._timestamps.popleft()

                gap_wait = 0.0
                if self._timestamps:
                    gap_wait = self.min_gap_seconds - (now - self._timestamps[-1])

                if len(self._timestamps) < self.max_requests and gap_wait <= 0:
                    self._timestamps.append(now)
                    return

                window_wait = 0.0
                if len(self._timestamps) >= self.max_requests:
                    window_wait = self.window_seconds - (now - self._timestamps[0]) + 0.05

                wait = max(gap_wait, window_wait, 0.05)

            time.sleep(wait)


_rate_limiter_1 = _RateLimiter(
    MISTRAL_MAX_REQUESTS_PER_MINUTE, min_gap_seconds=MISTRAL_MIN_REQUEST_GAP_SECONDS
)
_rate_limiter_2 = (
    _RateLimiter(MISTRAL_MAX_REQUESTS_PER_MINUTE_2, min_gap_seconds=MISTRAL_MIN_REQUEST_GAP_SECONDS)
    if MISTRAL_KEY_2
    else None
)

# Number of focal assets at/above which an event is split across the two
# keys instead of run sequentially on one. Below this, a single key
# comfortably fits inside prediction_deadline, so there's no reason to
# involve a second account. Only takes effect if MISTRAL_KEY_2 is set.
DUAL_KEY_THRESHOLD = int(os.getenv("DUAL_KEY_THRESHOLD", "4"))

# Fama-French 12 industry benchmarks: {code: (gross_margin_pct, roe_pct)}
# Aggregated from Wharton Research Data Services (WRDS).
FF12_BENCHMARKS: dict[str, tuple[float, float]] = {
    "BusEq": (52, 0),
    "Chems": (35, 0),
    "Durbl": (27, 2),
    "Enrgy": (38, 3),
    "Hlth": (45, -42),
    "Manuf": (32, 4),
    "Money": (55, 1),
    "NoDur": (39, 4),
    "Other": (34, -3),
    "Shops": (30, 5),
    "Telcm": (48, 2),
    "Utils": (38, 5),
}

# Industry used when classification fails outright after all retries.
FF12_FALLBACK = "Other"


def predict(event: dict) -> list[dict]:
    """Return predictions for one Explaining Markets event.

    `event` is the verified webhook payload. Useful fields:
      event["event_type"]          e.g. "EARNINGS_RELEASE"
      event["focal_assets"]        list of {"identifier_type", "identifier_value"}
      event["information_url"]     short-lived signed URL with the event summary JSON
      event["prediction_deadline"] ISO timestamp; submit before this fires

    Required return: a list of dicts, one per focal asset:
      [{"identifier_value": "AAPL", "predicted_percentile": 0.71}, ...]

    `predicted_percentile` is a float in [0, 1] — your prediction of where the
    asset's next-day *unexpected* return will fall in its historical distribution
    (0 = worst, 0.50 = median, 1 = best).
    """
    summary = httpx.get(event["information_url"], timeout=10.0)
    summary.raise_for_status()
    summary_json = summary.json()

    print(summary_json)

    assets = event["focal_assets"]
    event_type = event["event_type"]

    if len(assets) >= DUAL_KEY_THRESHOLD and client_2 is not None:
        print(
            f"[INFO] {len(assets)} focal assets >= DUAL_KEY_THRESHOLD "
            f"({DUAL_KEY_THRESHOLD}) and MISTRAL_KEY_2 is set — splitting "
            f"across both keys concurrently."
        )
        return _predict_dual_key(assets, event_type, summary_json)

    return _predict_single_key(assets, event_type, summary_json, client, _rate_limiter_1)


def _predict_single_key(
    assets: list[dict],
    event_type: str,
    summary_json: dict,
    mistral_client: Mistral,
    rate_limiter: _RateLimiter,
) -> list[dict]:
    """Run every asset sequentially on one Mistral key.

    Pacing between calls is handled entirely by `rate_limiter.acquire()`
    inside each individual Mistral call (see `_classify_ff12` etc.), not by
    a manual sleep here — that's what lets this correctly share the
    account's real budget with OTHER events processing concurrently on the
    same key, not just with itself.
    """
    predictions: list[dict] = []
    for asset in assets:
        predictions.append(
            {
                "identifier_value": asset["identifier_value"],
                "predicted_percentile": _ask_llm(
                    mistral_client=mistral_client,
                    rate_limiter=rate_limiter,
                    summary=summary_json,
                    ticker=asset["identifier_value"],
                    event_type=event_type,
                ),
            }
        )
    return predictions


def _predict_dual_key(
    assets: list[dict], event_type: str, summary_json: dict
) -> list[dict]:
    """Split focal assets roughly in half and run each half concurrently,
    each on its own Mistral key end-to-end (classify + predict + sanity for
    every asset in its half).

    Concurrency only happens ACROSS the two halves — each half internally
    stays fully sequential, gated by that key's own `_RateLimiter`, so
    neither key's own rate limit is ever at risk of being tripped by this
    code alone. Order of the returned list follows the original asset order
    (group A's results first, then group B's), so the split is invisible to
    the caller.
    """
    midpoint = (len(assets) + 1) // 2
    group_a, group_b = assets[:midpoint], assets[midpoint:]

    results: dict[str, list[dict]] = {}

    def run(group: list[dict], mistral_client: Mistral, rate_limiter: _RateLimiter, key: str) -> None:
        # Every Mistral-call failure is already caught and given a safe
        # fallback inside _classify_ff12 / _predict_percentile /
        # _sanity_check. This try/except is a last-resort backstop for
        # anything NOT anticipated by those (an SDK bug, an unexpected
        # response shape, etc.). Thread.join() does not propagate worker
        # exceptions, so without this, a crash here would silently drop
        # this entire half's predictions from the return value with no
        # error surfaced to the caller. Every other failure mode in this
        # file falls back to a safe placeholder instead of dropping data
        # -- this makes the dual-key path honor that same guarantee: the
        # returned list always has exactly one entry per input asset.
        try:
            results[key] = _predict_single_key(
                group, event_type, summary_json, mistral_client, rate_limiter
            )
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see comment above
            print(
                f"[WARN] Unhandled error in dual-key thread {key!r} "
                f"({[a['identifier_value'] for a in group]}): {exc!r} — "
                f"submitting 0.5 placeholder for this half."
            )
            results[key] = [
                {"identifier_value": asset["identifier_value"], "predicted_percentile": 0.5}
                for asset in group
            ]

    thread_a = threading.Thread(target=run, args=(group_a, client, _rate_limiter_1, "a"))
    thread_b = threading.Thread(target=run, args=(group_b, client_2, _rate_limiter_2, "b"))
    thread_a.start()
    thread_b.start()
    thread_a.join()
    thread_b.join()

    return results.get("a", []) + results.get("b", [])


FF12_CLASSIFY_PROMPT = """
Classify the company below into exactly one Fama-French 12 industry group.

Codes: BusEq, Chems, Durbl, Enrgy, Hlth, Manuf, Money, NoDur, Other, Shops, Telcm, Utils

BusEq = Business Equipment (computers, software, electronics)
Chems = Chemicals
Durbl = Consumer Durables (cars, appliances, furniture)
Enrgy = Energy (oil, gas, coal)
Hlth  = Healthcare, Medical Equipment, Pharma
Manuf = Manufacturing (machinery, industrial)
Money = Finance (banks, insurance, real estate)
NoDur = Consumer Non-Durables (food, tobacco, apparel)
Other = Mines, construction, transport, hotels, entertainment, everything else
Shops = Wholesale, Retail
Telcm = Telecommunications
Utils = Utilities

Ticker: {ticker}
Event type: {event_type}

Facts:
{facts}

Respond with ONLY the code (e.g., `Manuf`). No other text.
"""

BASE_PROMPT = """
You are a quantitative financial analyst. Predict the cross-sectional percentile (0.00-1.00) of a stock's unexpected return following an earnings call, INCREMENTAL TO EARNINGS SURPRISE.

Industry benchmark (Gross Margin %, ROE %): {ff12_benchmark}

RULES (apply all that fire; skip Revenue if Guidance fired). Where a range is given, scale within it by the magnitude of the underlying number: small/marginal moves near the low end, large moves near the high end.
1. Guidance: raised → +0.05 to +0.13 (larger % raise → higher end). lowered → -0.09 to -0.18 (larger % cut → more negative end).
2. Margin: GM ≥ benchmark+5pp → +0.05 to +0.09, scaling with how far above. GM ≤ benchmark-5pp → -0.05 to -0.09, scaling with how far below. "pricing power" cited → additional +0.02 to +0.05.
3. Revenue (skip if Guidance fired): growth → +0.01 to +0.04. decline → -0.03 to -0.08.
4. Weak language ("headwinds","pressure","slowing","uncertain","macro concerns"): -0.02 each, cap 3.
5. Ignore one-time items (asset sales, restructuring, impairments, tax items, FX).

FINAL_PREDICTION = clamp(0.50 + sum(fired deltas), 0.00, 1.00)
Respond with ONLY the number representing the predicted percentile (e.g., `0.62`).

Facts: {facts}
FINAL_PREDICTION:
"""

SANITY_CHECK_PROMPT = """
You are auditing a colleague's earnings-call return prediction for signs of hallucination.

Facts the prediction was based on:
{facts}

Colleague's predicted percentile (0.00-1.00, where 0.50 = median/no-surprise): {predicted_percentile}

Would a careful, literal reading of the facts above plausibly support a percentile in that range? Flag it ONLY if the percentile clearly contradicts the facts (e.g. it's near 1.00 despite lowered guidance or contracting margins, or near 0.00 despite raised guidance and expanding margins) or appears to rely on specific claims/numbers not present in the facts.

Respond with ONLY one word: OK or FLAG.
"""


def _is_rate_limit_error(exc: Exception) -> bool:
    """True if `exc` looks like a 429 from the Mistral SDK/HTTP layer.

    Checked so a 429 retry can skip the fixed RETRY_BACKOFF_SECONDS sleep —
    the next loop iteration's `rate_limiter.acquire()` already provides the
    correct wait, so an extra fixed sleep on top would just slow things down
    without adding any protection.
    """
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    text = str(exc).lower()
    return "429" in text or "rate_limited" in text or "rate limit" in text


def _ask_llm(
    *, mistral_client: Mistral, rate_limiter: _RateLimiter, summary: dict, ticker: str, event_type: str
) -> float:
    """Ask Mistral for a calibrated percentile for one asset, using the
    given client/key and rate limiter (so this works identically whether
    it's running on the single-key path or as one half of the dual-key
    path).

    Three Mistral calls happen here — see the module docstring for the
    full rundown of `_classify_ff12` / `_predict_percentile` / `_sanity_check`.
    Each of those calls acquires its own slot from `rate_limiter` right
    before it fires, so this is automatically correctly paced against
    every OTHER call sharing the same key — including calls from other
    events' `predict()` invocations running concurrently.
    If the main prediction call never produces a usable number, or the
    sanity check explicitly flags the number it did produce, this returns
    0.5 for that asset so one bad event never stalls the whole run.
    """
    summary_text = summary.get("summary") if isinstance(summary, dict) else None
    if not summary_text:
        summary_text = json.dumps(summary)
    summary_text = summary_text[:8000]

    facts = f"Event type: {event_type}\nTicker: {ticker}\n\n{summary_text}"

    industry = _classify_ff12(
        mistral_client=mistral_client,
        rate_limiter=rate_limiter,
        facts=summary_text,
        ticker=ticker,
        event_type=event_type,
    )
    gross_margin, roe = FF12_BENCHMARKS[industry]
    ff12_benchmark = f"{industry} — Gross Margin {gross_margin}%, ROE {roe}%"

    prompt = BASE_PROMPT.format(ff12_benchmark=ff12_benchmark, facts=facts)
    predicted_percentile = _predict_percentile(
        mistral_client=mistral_client, rate_limiter=rate_limiter, prompt=prompt, ticker=ticker
    )
    if predicted_percentile is None:
        print(
            f"[WARN] Mistral call failed for {ticker} after {MAX_ATTEMPTS} attempts "
            f"— submitting 0.5 placeholder."
        )
        return 0.5

    if not _sanity_check(
        mistral_client=mistral_client,
        rate_limiter=rate_limiter,
        facts=facts,
        ticker=ticker,
        predicted_percentile=predicted_percentile,
    ):
        print(
            f"[WARN] Sanity check flagged {ticker}'s predicted percentile "
            f"({predicted_percentile:.2f}) as inconsistent with the facts — "
            f"submitting 0.5 placeholder instead."
        )
        return 0.5

    return predicted_percentile


def _classify_ff12(
    *, mistral_client: Mistral, rate_limiter: _RateLimiter, facts: str, ticker: str, event_type: str
) -> str:
    """Classify the company into an FF12 industry via one Mistral call.

    Retries transient failures the same way the other calls in this module
    do. Falls back to FF12_FALLBACK ("Other") if every attempt fails, rather
    than raising — a wrong-but-plausible industry benchmark is preferable to
    stalling the whole prediction run over a classification hiccup.
    """
    prompt = FF12_CLASSIFY_PROMPT.format(ticker=ticker, event_type=event_type, facts=facts)

    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        rate_limiter.acquire()
        try:
            response = mistral_client.chat.complete(
                model=MISTRAL_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            return _parse_ff12(response.choices[0].message.content)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            last_error = exc
            if attempt < MAX_ATTEMPTS:
                print(
                    f"[WARN] Mistral FF12 classify failed for {ticker} "
                    f"(attempt {attempt}/{MAX_ATTEMPTS}): {exc!r}"
                )
                if not _is_rate_limit_error(exc):
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    print(
        f"[WARN] Mistral FF12 classify failed for {ticker} after {MAX_ATTEMPTS} "
        f"attempts ({last_error!r}) — falling back to {FF12_FALLBACK!r}."
    )
    return FF12_FALLBACK


def _predict_percentile(
    *, mistral_client: Mistral, rate_limiter: _RateLimiter, prompt: str, ticker: str
) -> float | None:
    """Run the main percentile-prediction call, with retries.

    Returns None — rather than the 0.5 placeholder — if every attempt fails,
    so the caller (`_ask_llm`) can decide the fallback itself and skip the
    sanity check on a number that was never actually predicted.
    """
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        rate_limiter.acquire()
        try:
            response = mistral_client.chat.complete(
                model=MISTRAL_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            return _parse_percentile(response.choices[0].message.content)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            last_error = exc
            if attempt < MAX_ATTEMPTS:
                print(
                    f"[WARN] Mistral call failed for {ticker} "
                    f"(attempt {attempt}/{MAX_ATTEMPTS}): {exc!r}"
                )
                if not _is_rate_limit_error(exc):
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    return None


def _sanity_check(
    *, mistral_client: Mistral, rate_limiter: _RateLimiter, facts: str, ticker: str, predicted_percentile: float
) -> bool:
    """Second-pass hallucination check: shown the same facts plus the
    percentile from `_predict_percentile`, does that percentile hold up?

    Returns True ("OK") on an explicit pass, or if every attempt at running
    the check itself fails — an unreachable audit call is not evidence that
    the original prediction was wrong, so a network hiccup here doesn't
    discard an otherwise-fine prediction. Returns False only on an explicit
    "FLAG" verdict from the model.
    """
    prompt = SANITY_CHECK_PROMPT.format(
        facts=facts, predicted_percentile=f"{predicted_percentile:.2f}"
    )

    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        rate_limiter.acquire()
        try:
            response = mistral_client.chat.complete(
                model=MISTRAL_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            return _parse_verdict(response.choices[0].message.content)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            last_error = exc
            if attempt < MAX_ATTEMPTS:
                print(
                    f"[WARN] Mistral sanity check failed for {ticker} "
                    f"(attempt {attempt}/{MAX_ATTEMPTS}): {exc!r}"
                )
                if not _is_rate_limit_error(exc):
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    print(
        f"[WARN] Mistral sanity check failed for {ticker} after {MAX_ATTEMPTS} "
        f"attempts ({last_error!r}) — leaving prediction unchanged."
    )
    return True


def _parse_percentile(content: str | None) -> float:
    """Parse the model's raw reply into a float clamped to [0, 1].

    Raises ValueError if no number can be found, so `_predict_percentile`'s
    retry loop treats an unparseable reply the same as a failed API call.
    """
    if not content:
        raise ValueError("empty response from Mistral")

    match = re.search(r"-?\d*\.?\d+", content.strip())
    if not match:
        raise ValueError(f"could not find a number in Mistral response: {content!r}")

    return max(0.0, min(1.0, float(match.group())))


def _parse_ff12(content: str | None) -> str:
    """Parse the model's raw reply into one of the FF12_BENCHMARKS keys.

    Raises ValueError if no known code is found, so `_classify_ff12`'s retry
    loop treats an unparseable reply the same as a failed API call.
    """
    if not content:
        raise ValueError("empty response from Mistral")

    text = content.strip()
    for code in FF12_BENCHMARKS:
        if re.search(rf"\b{code}\b", text, re.IGNORECASE):
            return code

    raise ValueError(f"could not find a known FF12 code in Mistral response: {content!r}")


def _parse_verdict(content: str | None) -> bool:
    """Parse the sanity-check reply into True (OK / pass) or False (FLAG).

    FLAG is checked before OK since a hedged reply like "FLAG - not OK"
    should still count as a flag. Raises ValueError on anything unrecognized,
    so `_sanity_check`'s retry loop treats it the same as a failed API call.
    """
    if not content:
        raise ValueError("empty response from Mistral")

    text = content.strip().upper()
    if "FLAG" in text:
        return False
    if "OK" in text:
        return True

    raise ValueError(f"could not find OK/FLAG in Mistral sanity check response: {content!r}")