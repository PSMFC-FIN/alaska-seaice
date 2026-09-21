"""
Tests for download_from_nsidc.py

Two kinds of checks, split so CI/offline runs still get signal:

1. Offline / mocked tests (always run) — verify the URL-building, regex
   and dry-run "what would be downloaded" logic without touching the
   network. These catch bugs like miscounted years or bad URL joins.

2. Live network tests (auto-skipped if NSIDC isn't reachable) — actually
   hit the NSIDC HTTPS server to confirm the source URLs are still valid
   and preview the real filenames that would be pulled down, without
   downloading any data files.

Run everything, with the preview output visible:
    pytest -v -s test_download_from_nsidc.py

Run only the offline tests (no network needed):
    pytest -v -m "not network" test_download_from_nsidc.py
"""
from datetime import date
from urllib.parse import urlparse

import pytest
import requests

import download_from_nsidc as dl


# ---------------------------------------------------------------------------
# Network availability gate
# ---------------------------------------------------------------------------

def _nsidc_reachable() -> bool:
    # A raw TCP connect can succeed even when an HTTP(S) proxy or egress
    # policy rejects the actual request, so probe with a real request
    # (this is also exactly what the code under test does).
    try:
        requests.head(dl.CDR_BASE, timeout=8)
        return True
    except requests.RequestException:
        return False


NSIDC_UP = _nsidc_reachable()
network = pytest.mark.skipif(
    not NSIDC_UP, reason="noaadata.apps.nsidc.org not reachable from this environment"
)


# ---------------------------------------------------------------------------
# 1. Offline structural checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [dl.CDR_BASE, dl.NRT_BASE])
def test_base_url_is_well_formed_https(url):
    parsed = urlparse(url)
    assert parsed.scheme == "https"
    assert parsed.netloc == "noaadata.apps.nsidc.org"
    assert url.endswith("/"), "base URL must end in / so year/filename joins stay valid"


def test_nrt_year_range_is_actually_defined():
    """
    Regression guard for a real bug: NRT_YEAR_RANGE is commented out
    (script line ~53) but main() passes it to download_product() for the
    NRT branch, which runs by default (do_nrt is True unless --cdr-only
    is passed). As shipped, `python download_from_nsidc.py` and
    `python download_from_nsidc.py --nrt-only` both crash with
    NameError: name 'NRT_YEAR_RANGE' is not defined.
    """
    assert hasattr(dl, "NRT_YEAR_RANGE"), (
        "NRT_YEAR_RANGE is not defined at module level. Uncomment "
        "`NRT_YEAR_RANGE = (2025, 2025)` (or set the real bounds) — "
        "otherwise any run that includes NRT will raise NameError."
    )


def test_cdr_year_range_covers_current_year():
    first, last = dl.CDR_YEAR_RANGE
    assert first <= date.today().year <= last, (
        f"CDR_YEAR_RANGE={dl.CDR_YEAR_RANGE} does not include "
        f"{date.today().year} — the docstring claims CDR is 1978-present, "
        "but the range is hardcoded to a single year, so it must be "
        "bumped by hand each year."
    )


def test_file_pattern_matches_expected_naming_and_rejects_junk():
    assert dl.FILE_PATTERN.search("sic_psn25_20260101_F18_v05r00.nc")
    assert not dl.FILE_PATTERN.search("readme.txt")
    assert not dl.FILE_PATTERN.search("sic_psn25_2026_summary.csv")


def test_list_remote_files_double_counts_every_entry(monkeypatch):
    """
    Regression guard for a real bug: a standard Apache-style directory
    listing repeats each filename twice per row — once in the href
    attribute, once as the visible link text:

        <a href="sic_psn25_20260101_F18_v05r00.nc">sic_psn25_20260101_F18_v05r00.nc</a>

    FILE_PATTERN.findall() runs against the *whole* HTML text, so it
    matches both occurrences. list_remote_files() therefore returns each
    real filename twice, which inflates the "files it plans to grab"
    counts reported by download_product() (e.g. "24 files (12 new)" for
    a day that only has 12). It isn't destructive — download_file()'s
    existence check means the second hit per file is a no-op — but the
    reported counts and the download loop's work are 2x what they should
    be. If this starts failing, either the script was fixed to dedupe
    (great, update this test) or NSIDC's listing format changed.
    """
    sample_html = """
    <html><body>
    <a href="sic_psn25_20260101_F18_v05r00.nc">sic_psn25_20260101_F18_v05r00.nc</a>
    <a href="sic_psn25_20260102_F18_v05r00.nc">sic_psn25_20260102_F18_v05r00.nc</a>
    <a href="../">../</a>
    </body></html>
    """

    class FakeResponse:
        status_code = 200
        text = sample_html

        def raise_for_status(self):
            pass

    monkeypatch.setattr(requests, "get", lambda url, timeout=30: FakeResponse())

    files = dl.list_remote_files(dl.CDR_BASE, 2026)
    assert files.count("sic_psn25_20260101_F18_v05r00.nc") == 2
    assert files.count("sic_psn25_20260102_F18_v05r00.nc") == 2
    assert len(files) == 4  # would be 2 if the script deduped


def test_list_remote_files_returns_empty_list_on_request_failure(monkeypatch, capsys):
    def raise_it(url, timeout=30):
        raise requests.exceptions.ConnectionError("simulated outage")

    monkeypatch.setattr(requests, "get", raise_it)

    files = dl.list_remote_files(dl.CDR_BASE, 2026)
    assert files == []
    assert "WARNING" in capsys.readouterr().out


def test_list_remote_files_returns_empty_list_on_http_error(monkeypatch):
    class FakeResponse:
        status_code = 404
        text = "not found"

        def raise_for_status(self):
            raise requests.exceptions.HTTPError("404")

    monkeypatch.setattr(requests, "get", lambda url, timeout=30: FakeResponse())
    assert dl.list_remote_files(dl.CDR_BASE, 1900) == []


def test_download_product_dry_run_builds_expected_urls_and_paths(monkeypatch, tmp_path):
    """
    Full "what would be grabbed" preview, entirely offline: fake out
    list_remote_files and download_file, then check download_product
    builds the exact URLs and destination paths a real run would use,
    across a multi-year range with start/end clamping.
    """
    planned = []

    def fake_list_remote_files(base_url, year):
        return [f"sic_psn25_{year}0101_F18_v05r00.nc", f"sic_psn25_{year}0102_F18_v05r00.nc"]

    def fake_download_file(url, dest_path):
        planned.append((url, dest_path))

    monkeypatch.setattr(dl, "list_remote_files", fake_list_remote_files)
    monkeypatch.setattr(dl, "download_file", fake_download_file)

    dl.download_product(
        dl.CDR_BASE, (2018, 2026), tmp_path, start_year=2024, end_year=2025
    )

    urls = [u for u, _ in planned]
    assert urls == [
        f"{dl.CDR_BASE}2024/sic_psn25_20240101_F18_v05r00.nc",
        f"{dl.CDR_BASE}2024/sic_psn25_20240102_F18_v05r00.nc",
        f"{dl.CDR_BASE}2025/sic_psn25_20250101_F18_v05r00.nc",
        f"{dl.CDR_BASE}2025/sic_psn25_20250102_F18_v05r00.nc",
    ]
    dests = [d for _, d in planned]
    assert dests == [
        tmp_path / "2024" / "sic_psn25_20240101_F18_v05r00.nc",
        tmp_path / "2024" / "sic_psn25_20240102_F18_v05r00.nc",
        tmp_path / "2025" / "sic_psn25_20250101_F18_v05r00.nc",
        tmp_path / "2025" / "sic_psn25_20250102_F18_v05r00.nc",
    ]


def test_download_product_skips_years_outside_clamped_range(monkeypatch, tmp_path):
    seen_years = []

    def fake_list_remote_files(base_url, year):
        seen_years.append(year)
        return []

    monkeypatch.setattr(dl, "list_remote_files", fake_list_remote_files)
    dl.download_product(dl.CDR_BASE, (2020, 2026), tmp_path, start_year=2024, end_year=2024)
    assert seen_years == [2024]


# ---------------------------------------------------------------------------
# 2. Live checks against the real NSIDC server
# ---------------------------------------------------------------------------

@network
@pytest.mark.parametrize("url", [dl.CDR_BASE, dl.NRT_BASE])
def test_source_base_url_is_reachable(url):
    resp = requests.get(url, timeout=30)
    assert resp.status_code == 200, f"{url} returned HTTP {resp.status_code}"


@network
def test_cdr_year_directory_is_reachable_and_lists_files(capsys):
    year = dl.CDR_YEAR_RANGE[0]
    url = f"{dl.CDR_BASE}{year}/"
    resp = requests.get(url, timeout=30)
    assert resp.status_code == 200, f"{url} returned HTTP {resp.status_code}"

    files = dl.list_remote_files(dl.CDR_BASE, year)
    assert files, (
        f"{url} is reachable but FILE_PATTERN matched 0 files — either "
        "the year has no data yet or FILE_PATTERN no longer matches "
        "NSIDC's current naming convention."
    )
    print(f"\nCDR {year}: {len(files)} files would be grabbed from {url}")
    for f in files[:5]:
        print(f"   {f}")
    if len(files) > 5:
        print(f"   ... and {len(files) - 5} more")


@network
def test_nrt_year_directory_is_reachable(capsys):
    year = date.today().year
    url = f"{dl.NRT_BASE}{year}/"
    resp = requests.get(url, timeout=30)
    assert resp.status_code == 200, f"{url} returned HTTP {resp.status_code}"

    files = dl.list_remote_files(dl.NRT_BASE, year)
    print(f"\nNRT {year}: {len(files)} files would be grabbed from {url}")
    for f in files[:5]:
        print(f"   {f}")
    if len(files) > 5:
        print(f"   ... and {len(files) - 5} more")
    # NRT for the current year can legitimately be empty very early in
    # January, so this is a warning rather than a hard assertion.
    if not files:
        print("   WARNING: 0 files matched — verify manually if this is unexpected.")