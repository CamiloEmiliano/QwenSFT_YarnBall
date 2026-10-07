"""Tests for vast_runner automation and security constraints."""

from unittest.mock import patch
import pytest

from vast_runner import (
    search_offers,
    format_offers_table,
    main,
)


def test_search_offers_enforces_datacenter_and_verified():
    """Verify that search_offers query string always contains datacenter=true and verified=true."""
    captured_args = []

    def mock_run_vast_cmd(cmd_args, raw_json=True):
        captured_args.append(cmd_args)
        return []

    with patch("vast_runner.run_vast_cmd", side_effect=mock_run_vast_cmd):
        search_offers(gpu_name="A100_SXM4", min_vram=40, max_dph=2.0)

    assert len(captured_args) == 1
    query = captured_args[0][2]
    assert "datacenter=true" in query, "Security violation: datacenter=true must be enforced"
    assert "verified=true" in query, "Security violation: verified=true must be enforced"
    assert "gpu_ram >= 40" in query
    assert "dph_total <= 2.0" in query


def test_format_offers_table_vram_gb_conversion():
    """Verify that raw VRAM in MB (e.g. 81920) is cleanly converted to GB (80 GB)."""
    sample_offers = [
        {
            "id": 1234567,
            "gpu_name": "A100 SXM4",
            "gpu_ram": 81920.0,
            "dph_total": 1.07,
            "inet_down": 2500.0,
            "country": "US",
            "reliability2": 0.998,
        },
        {
            "id": 7654321,
            "gpu_name": "RTX A5000",
            "gpu_ram": 24564.0,
            "dph_total": 0.23,
            "inet_down": 850.0,
            "country": "EU",
            "reliability2": 0.995,
        },
    ]

    table = format_offers_table(sample_offers)
    assert "1234567" in table
    assert "A100 SXM4" in table
    assert "80 GB" in table, "81920 MB should display as 80 GB"
    assert "$1.07" in table
    assert "99.8%" in table

    assert "7654321" in table
    assert "24 GB" in table, "24564 MB should display as 24 GB"
    assert "$0.23" in table


def test_format_offers_table_empty():
    """Verify fallback message when no offers match."""
    table = format_offers_table([])
    assert "No matching verified datacenter instances found" in table


def test_cli_search_dry_run_dispatch():
    """Verify CLI argument parsing for 'launch --dry-run'."""
    with patch("sys.argv", ["vast_runner.py", "launch", "--gpu", "A100_SXM4", "--dry-run"]):
        with patch("vast_runner.search_offers", return_value=[{"id": 9999, "gpu_name": "A100", "dph_total": 1.0}]):
            # Should run without error
            main()
