import base64
import sqlite3
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

st.set_page_config(page_title="Companies Incorporated Today", layout="wide")

# ── Constants ────────────────────────────────────────────────────────────────

TECH_SIC_CODES = frozenset({
    "62012", "62020", "58290", "58210", "61100", "61200", "61300", "61900",
    "62011", "62030", "62090", "63110", "63120", "71200", "72110", "72190",
    "72200", "71129",
})

HOLDINGS_SIC_CODES = frozenset({
    "64201", "64202", "64203", "64204", "64205", "64209", "66300",
})

TARGET_SIC_CODES = sorted(TECH_SIC_CODES | HOLDINGS_SIC_CODES)
TARGET_SIC_SET = TECH_SIC_CODES | HOLDINGS_SIC_CODES  # O(1) membership tests

EXCLUDED_DIRECTOR_COUNTRIES = frozenset({
    "PAKISTAN", "TURKEY", "CHINA", "NIGERIA",
})

COUNTRY_NORMALISATION = {
    "TURKIYE": "TURKEY",
    "PEOPLE'S REPUBLIC OF CHINA": "CHINA",
    "PRC": "CHINA",
    "P.R.C.": "CHINA",
}

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "companies_cache.db"
APP_USER_AGENT = "streamlit-companies-house-today-app"
REQUEST_TIMEOUT = 30
ADVANCED_SEARCH_PAGE_SIZE = 1000
MAX_WORKERS = 10          # raised — I/O-bound; tune to key count × 3–4
RATE_WINDOW_SECONDS = 300
SAFE_REQUESTS_PER_WINDOW = 540

# Pre-compute auth headers once per key rather than on every request
_AUTH_HEADER_CACHE: Dict[str, Dict[str, str]] = {}
_AUTH_CACHE_LOCK = threading.Lock()

# ── Per-key rate limiter using a deque (O(1) append/popleft) ─────────────────
_rate_buckets: Dict[str, deque] = {}
_rate_lock = threading.Lock()


def _get_auth_header(api_key: str) -> Dict[str, str]:
    """Cache Base64-encoded auth headers — encoding is pure CPU waste if repeated."""
    with _AUTH_CACHE_LOCK:
        if api_key not in _AUTH_HEADER_CACHE:
            token = base64.b64encode(f"{api_key}:".encode()).decode()
            _AUTH_HEADER_CACHE[api_key] = {
                "Authorization": f"Basic {token}",
                "User-Agent": APP_USER_AGENT,
            }
        return _AUTH_HEADER_CACHE[api_key]


def today_uk_str() -> str:
    return datetime.now().astimezone().date().isoformat()


def now_uk_str() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")


def get_api_keys() -> List[str]:
    keys: List[str] = []
    list_style_keys = st.secrets.get("COMPANIES_HOUSE_API_KEYS", [])
    if list_style_keys:
        keys.extend([str(k).strip() for k in list_style_keys if str(k).strip()])
    for key_name in ["CH_API_KEY_1", "CH_API_KEY_2", "CH_API_KEY_3"]:
        value = st.secrets.get(key_name, "")
        if value:
            keys.append(str(value).strip())
    seen: set = set()
    return [k for k in keys if k and not (k in seen or seen.add(k))]  # one-pass dedup


def classify_sector(sic_codes: List[str]) -> Optional[str]:
    codes = set(sic_codes) if sic_codes else set()
    if codes & HOLDINGS_SIC_CODES:
        return "Holdings"
    if codes & TECH_SIC_CODES:
        return "Tech"
    return None


def normalise_country(country: str) -> str:
    value = country.strip().upper() if country else ""
    return COUNTRY_NORMALISATION.get(value, value) if value else ""


# ── HTTP session ──────────────────────────────────────────────────────────────

@st.cache_resource
def get_http_session() -> requests.Session:
    retry_strategy = Retry(
        total=4,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    # Pool size matches MAX_WORKERS so no connection is ever queued waiting
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=MAX_WORKERS,
        pool_maxsize=MAX_WORKERS + 4,
    )
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ── SQLite ────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    # WAL mode + aggressive memory use — reads never block writes
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-32768;")   # 32 MB page cache (was default 2 MB)
    conn.execute("PRAGMA mmap_size=268435456;") # 256 MB memory-mapped I/O
    conn.execute("""
        CREATE TABLE IF NOT EXISTS director_country_cache (
            company_number TEXT PRIMARY KEY,
            excluded INTEGER NOT NULL,
            director_countries TEXT,
            checked_at TEXT NOT NULL
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_company_snapshots (
            run_date TEXT NOT NULL,
            company_number TEXT NOT NULL,
            company_name TEXT NOT NULL,
            sector TEXT NOT NULL,
            time_added_to_table TEXT NOT NULL,
            pull_order INTEGER NOT NULL,
            PRIMARY KEY (run_date, company_number)
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_seen_companies (
            run_date TEXT NOT NULL,
            company_number TEXT NOT NULL,
            PRIMARY KEY (run_date, company_number)
        )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshot_run_date ON daily_company_snapshots(run_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_seen_run_date ON daily_seen_companies(run_date)")
    conn.commit()
    return conn


# ── Rate limiter (deque-based, O(1)) ─────────────────────────────────────────

def throttle_for_key(api_key: str) -> None:
    """Sliding-window rate limiter using collections.deque for O(1) operations."""
    while True:
        now_ts = time.monotonic()  # monotonic avoids DST/NTP jumps
        with _rate_lock:
            bucket = _rate_buckets.setdefault(api_key, deque())
            cutoff = now_ts - RATE_WINDOW_SECONDS
            while bucket and bucket[0] < cutoff:
                bucket.popleft()          # O(1) vs list.pop(0) which is O(n)
            if len(bucket) < SAFE_REQUESTS_PER_WINDOW:
                bucket.append(now_ts)
                return
            sleep_for = max(0.25, RATE_WINDOW_SECONDS - (now_ts - bucket[0]))
        time.sleep(min(sleep_for, 2.0))


def fetch_with_rotation(
    session: requests.Session,
    url: str,
    params: Optional[Dict],
    api_keys: List[str],
    timeout: int = REQUEST_TIMEOUT,
) -> requests.Response:
    last_response = None
    for api_key in api_keys:
        throttle_for_key(api_key)
        response = session.get(url, headers=_get_auth_header(api_key), params=params, timeout=timeout)
        if response.status_code in (401, 429):
            last_response = response
            continue
        response.raise_for_status()
        return response
    if last_response is not None:
        last_response.raise_for_status()
    raise RuntimeError("No valid Companies House API keys were available.")


# ── DB helpers ────────────────────────────────────────────────────────────────

_write_lock = threading.Lock()


def get_cached_decisions_map(company_numbers: List[str]) -> Dict[str, bool]:
    if not company_numbers:
        return {}
    conn = get_db_connection()
    placeholders = ",".join("?" * len(company_numbers))
    rows = conn.execute(
        f"SELECT company_number, excluded FROM director_country_cache WHERE company_number IN ({placeholders})",
        company_numbers,
    ).fetchall()
    return {cn: bool(ex) for cn, ex in rows}


def upsert_director_decision(company_number: str, excluded: bool, countries: List[str]) -> None:
    conn = get_db_connection()
    countries_str = ", ".join(sorted(set(countries)))
    checked_at = now_uk_str()
    with _write_lock:
        conn.execute(
            """
            INSERT INTO director_country_cache (company_number, excluded, director_countries, checked_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(company_number) DO UPDATE SET
                excluded = excluded,
                director_countries = excluded.director_countries,
                checked_at = excluded.checked_at
            """,
            (company_number, int(excluded), countries_str, checked_at),
        )
        conn.commit()


# ── Batch DB write: single transaction for all uncached results ───────────────

def batch_upsert_decisions(results: List[Tuple[str, bool, List[str]]]) -> None:
    """Write all officer-check results in ONE transaction — drastically faster than
    one commit per company when handling dozens of new companies."""
    if not results:
        return
    conn = get_db_connection()
    checked_at = now_uk_str()
    rows = [
        (cn, int(ex), ", ".join(sorted(set(countries))), checked_at)
        for cn, ex, countries in results
    ]
    with _write_lock:
        conn.executemany(
            """
            INSERT INTO director_country_cache (company_number, excluded, director_countries, checked_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(company_number) DO UPDATE SET
                excluded = excluded,
                director_countries = excluded.director_countries,
                checked_at = excluded.checked_at
            """,
            rows,
        )
        conn.commit()


def get_active_director_countries(
    session: requests.Session,
    company_number: str,
    api_keys: List[str],
) -> List[str]:
    url = f"https://api.company-information.service.gov.uk/company/{company_number}/officers"
    params = {
        "register_view": "true",
        "register_type": "directors",
        "items_per_page": "100",
    }
    response = fetch_with_rotation(session, url, params, api_keys)
    items = response.json().get("items") or []
    return [
        c for officer in items
        if (c := normalise_country(officer.get("country_of_residence", "")))
    ]


def check_company_exclusion(
    company_number: str,
    api_keys: List[str],
) -> Tuple[str, bool, List[str]]:
    """Returns (company_number, excluded, countries) — does NOT write to DB.
    Batch writing is handled by the caller for efficiency."""
    session = get_http_session()
    try:
        countries = get_active_director_countries(session, company_number, api_keys)
        excluded = any(c in EXCLUDED_DIRECTOR_COUNTRIES for c in countries)
    except requests.RequestException:
        countries = []
        excluded = False
    return company_number, excluded, countries


def fetch_candidate_companies(api_keys: List[str], run_date: str) -> pd.DataFrame:
    session = get_http_session()
    url = "https://api.company-information.service.gov.uk/advanced-search/companies"
    rows: List[dict] = []
    start_index = 0

    while True:
        params = {
            "incorporated_from": run_date,
            "incorporated_to": run_date,
            "sic_codes": ",".join(TARGET_SIC_CODES),
            "size": str(ADVANCED_SEARCH_PAGE_SIZE),
            "start_index": str(start_index),
        }
        response = fetch_with_rotation(session, url, params, api_keys)
        payload = response.json()
        items = payload.get("items") or []

        for item in items:
            company_number = item.get("company_number", "").strip()
            if not company_number:
                continue
            sic_codes = [str(c) for c in (item.get("sic_codes") or []) if c]
            sector = classify_sector(sic_codes)
            if sector:
                rows.append({
                    "company_number": company_number,
                    "company_name": item.get("company_name", ""),
                    "sector": sector,
                })

        if len(items) < ADVANCED_SEARCH_PAGE_SIZE:
            break
        start_index += ADVANCED_SEARCH_PAGE_SIZE

    if not rows:
        return pd.DataFrame(columns=["company_number", "company_name", "sector"])
    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=["company_number"], keep="first").reset_index(drop=True)


def enrich_exclusions(
    candidate_df: pd.DataFrame,
    api_keys: List[str],
) -> Tuple[pd.DataFrame, int, int]:
    if candidate_df.empty:
        return candidate_df.copy(), 0, 0

    company_numbers = candidate_df["company_number"].astype(str).tolist()
    cached_map = get_cached_decisions_map(company_numbers)
    cache_hits = len(cached_map)
    uncached_numbers = [n for n in company_numbers if n not in cached_map]

    if uncached_numbers:
        workers = min(MAX_WORKERS, len(uncached_numbers))
        fresh_results: List[Tuple[str, bool, List[str]]] = []

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(check_company_exclusion, cn, api_keys): cn
                for cn in uncached_numbers
            }
            for future in as_completed(futures):
                result = future.result()
                fresh_results.append(result)
                cached_map[result[0]] = result[1]

        # Single transaction for all new results — replaces N individual commits
        batch_upsert_decisions(fresh_results)

    checked_via_api = len(uncached_numbers)
    mask = candidate_df["company_number"].astype(str).map(cached_map).fillna(False)
    return candidate_df[~mask].reset_index(drop=True), checked_via_api, cache_hits


def load_snapshot(run_date: str) -> pd.DataFrame:
    conn = get_db_connection()
    df = pd.read_sql_query(
        "SELECT company_number, company_name, sector, time_added_to_table, pull_order "
        "FROM daily_company_snapshots WHERE run_date = ? ORDER BY pull_order DESC",
        conn,
        params=(run_date,),
    )
    if df.empty:
        return pd.DataFrame(columns=["company_number", "company_name", "sector", "time_added_to_table", "pull_order"])
    df["time_added_to_table"] = pd.to_datetime(df["time_added_to_table"], errors="coerce")
    df["pull_order"] = pd.to_numeric(df["pull_order"], errors="coerce")
    return df


def load_seen(run_date: str) -> pd.DataFrame:
    conn = get_db_connection()
    return pd.read_sql_query(
        "SELECT company_number FROM daily_seen_companies WHERE run_date = ?",
        conn,
        params=(run_date,),
    )


def identify_new_rows(current_df: pd.DataFrame, seen_df: pd.DataFrame) -> pd.DataFrame:
    if current_df.empty:
        return current_df.copy()
    if seen_df.empty or "company_number" not in seen_df.columns:
        return current_df.copy()
    seen_set = set(seen_df["company_number"].astype(str))   # O(1) lookup set
    return current_df[~current_df["company_number"].isin(seen_set)].reset_index(drop=True)


def save_daily_state(run_date: str, current_df: pd.DataFrame) -> None:
    conn = get_db_connection()
    export_df = current_df.copy()
    export_df["run_date"] = run_date
    export_df["time_added_to_table"] = export_df["time_added_to_table"].astype(str)
    with _write_lock:
        # Combine both DELETEs + both INSERTs into one transaction
        conn.execute("BEGIN")
        conn.execute("DELETE FROM daily_company_snapshots WHERE run_date = ?", (run_date,))
        conn.execute("DELETE FROM daily_seen_companies WHERE run_date = ?", (run_date,))
        export_df[
            ["run_date", "company_number", "company_name", "sector", "time_added_to_table", "pull_order"]
        ].to_sql("daily_company_snapshots", conn, if_exists="append", index=False)
        export_df[["run_date", "company_number"]].drop_duplicates().to_sql(
            "daily_seen_companies", conn, if_exists="append", index=False
        )
        conn.execute("COMMIT")


def get_cache_size() -> int:
    row = get_db_connection().execute("SELECT COUNT(*) FROM director_country_cache").fetchone()
    return int(row[0]) if row else 0


def render_table(df: pd.DataFrame, title: str) -> None:
    st.subheader(title)
    if df.empty:
        st.info("No companies to show yet.")
        return
    display_df = (
        df.sort_values("time_added_to_table", ascending=False, kind="stable")
        [["company_name", "sector", "time_added_to_table"]]
        .rename(columns={
            "company_name": "Company Name",
            "sector": "Sector",
            "time_added_to_table": "Time Added To Table",
        })
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)


@st.cache_data
def convert_df_to_download_csv(df: pd.DataFrame) -> bytes:
    return (
        df.sort_values("time_added_to_table", ascending=False, kind="stable")
        [["company_name", "sector", "time_added_to_table"]]
        .rename(columns={
            "company_name": "Company Name",
            "sector": "Sector",
            "time_added_to_table": "Time Added To Table",
        })
        .to_csv(index=False)
        .encode("utf-8")
    )


def build_current_day_dataset(
    api_keys: List[str],
    run_date: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, int, int]:
    candidate_df = fetch_candidate_companies(api_keys, run_date)
    filtered_df, checked_via_api, cache_hits = enrich_exclusions(candidate_df, api_keys)

    existing_df = load_snapshot(run_date)
    if existing_df.empty:
        current_df = filtered_df.copy()
        current_df["time_added_to_table"] = now_uk_str()
        current_df["pull_order"] = range(len(current_df))
    else:
        existing_numbers = set(existing_df["company_number"].astype(str))
        new_rows = filtered_df[~filtered_df["company_number"].astype(str).isin(existing_numbers)].copy()
        if not new_rows.empty:
            new_rows["time_added_to_table"] = now_uk_str()
            new_rows["pull_order"] = range(len(new_rows))
        current_df = (
            pd.concat([new_rows, existing_df], ignore_index=True)
            .drop_duplicates(subset=["company_number"], keep="first")
            .reset_index(drop=True)
        )

    seen_df = load_seen(run_date)
    new_df = identify_new_rows(current_df, seen_df)
    save_daily_state(run_date, current_df)
    return current_df, new_df, checked_via_api, cache_hits


def main() -> None:
    st.title("Companies Incorporated Today")
    st.caption(
        "Shows companies incorporated today matching Tech and Holdings SIC codes, "
        "excluding companies with active directors resident in Pakistan, Turkey, China, or Nigeria."
    )

    api_keys = get_api_keys()
    if not api_keys:
        st.error("Add COMPANIES_HOUSE_API_KEYS or CH_API_KEY_1/2/3 to your Streamlit secrets.")
        st.stop()

    run_date = today_uk_str()

    st.sidebar.header("Controls")
    st.sidebar.write(f"Run date: {run_date}")
    st.sidebar.write(f"API keys loaded: {len(api_keys)}")
    refresh = st.sidebar.button("Refresh now", type="primary")

    if refresh or "latest_df" not in st.session_state:
        started = time.perf_counter()
        current_df, new_df, checked_via_api, cache_hits = build_current_day_dataset(api_keys, run_date)
        elapsed = time.perf_counter() - started

        st.session_state.update({
            "latest_df": current_df,
            "new_df": new_df,
            "checked_this_run": checked_via_api,
            "cache_hits_this_run": cache_hits,
            "cache_size": get_cache_size(),
            "elapsed_seconds": round(elapsed, 2),
            "last_refresh": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        })
    else:
        current_df = load_snapshot(run_date)
        st.session_state.setdefault("latest_df", current_df)
        empty_cols = current_df.columns if not current_df.empty else ["company_number", "company_name", "sector", "time_added_to_table", "pull_order"]
        st.session_state.setdefault("new_df", pd.DataFrame(columns=empty_cols))
        st.session_state.setdefault("checked_this_run", 0)
        st.session_state.setdefault("cache_hits_this_run", 0)
        st.session_state.setdefault("cache_size", get_cache_size())
        st.session_state.setdefault("elapsed_seconds", 0.0)
        st.session_state.setdefault("last_refresh", "Not refreshed in this session")

    current_df = st.session_state.get("latest_df", pd.DataFrame())
    new_df = st.session_state.get("new_df", pd.DataFrame())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total pulled today", int(len(current_df)))
    c2.metric("New on latest refresh", int(len(new_df)))
    c3.metric("Officer checks this run", int(st.session_state.get("checked_this_run", 0)))
    c4.metric("Cache hits this run", int(st.session_state.get("cache_hits_this_run", 0)))
    c5.metric("Refresh seconds", float(st.session_state.get("elapsed_seconds", 0.0)))

    st.write(f"Last refresh: {st.session_state.get('last_refresh', 'Unknown')}")
    st.write(f"Persistent cached company decisions: {int(st.session_state.get('cache_size', 0))}")

    render_table(new_df, "New companies found on the latest refresh")
    render_table(current_df, "All companies pulled so far today")

    if not current_df.empty:
        st.download_button(
            label="Download today's results as CSV",
            data=convert_df_to_download_csv(current_df),
            file_name=f"companies_incorporated_{run_date}.csv",
            mime="text/csv",
            key="download_csv_button",
        )

    with st.expander("Suggested .streamlit/secrets.toml"):
        st.code(
            'COMPANIES_HOUSE_API_KEYS = [\n  "your-first-key",\n  "your-second-key",\n  "your-third-key"\n]',
            language="toml",
        )

    with st.expander("Architecture notes"):
        st.markdown("""
- Persistent SQLite caching with 32 MB page cache and 256 MB mmap for near-zero read latency.
- All officer-check results batched into a **single SQLite transaction** per refresh.
- Auth headers pre-computed and cached — Base64 encoding runs once per key, ever.
- `deque`-based sliding-window rate limiter: O(1) vs the original O(n) `list.pop(0)`.
- `time.monotonic()` for rate-limiting timestamps — immune to clock changes.
- `frozenset` for SIC code lookups — slightly faster membership than `set`.
- Thread pool sized to `MAX_WORKERS` with matching HTTP connection pool — no queueing.
- `save_daily_state` wraps all four DB operations in one explicit transaction.
- `session_state.update({})` batches state writes in a single call.
        """)


if __name__ == "__main__":
    main()
