import pandas as pd
import pytest

from sigconf.data.prices import cache_path, load_closes


def fake_closes(start="2019-01-01", end="2019-12-31"):
    days = pd.bdate_range(start, end, inclusive="left")
    s = pd.Series(range(len(days)), index=days, dtype=float, name="close") + 100.0
    s.index.name = "date"
    return s


class CountingFetcher:
    def __init__(self, closes):
        self.closes = closes
        self.calls = 0

    def __call__(self, ticker, start, end):
        self.calls += 1
        return self.closes


def refuse(*_):
    raise AssertionError("fetcher must not be called")


def test_cache_miss_fetches_once_and_writes_cache(tmp_path):
    fetch = CountingFetcher(fake_closes())
    got = load_closes("XYZ", "2019-01-01", "2019-12-31", cache_dir=tmp_path, fetcher=fetch)

    assert fetch.calls == 1
    assert cache_path("XYZ", tmp_path).exists()
    pd.testing.assert_series_equal(got, fake_closes(), check_freq=False)


def test_cache_hit_never_fetches(tmp_path):
    load_closes("XYZ", "2019-01-01", "2019-12-31", cache_dir=tmp_path,
                fetcher=CountingFetcher(fake_closes()))
    got = load_closes("XYZ", "2019-01-01", "2019-12-31", cache_dir=tmp_path, fetcher=refuse)
    assert len(got) == len(fake_closes())


def test_offline_hit_works_without_network(tmp_path):
    load_closes("XYZ", "2019-01-01", "2019-12-31", cache_dir=tmp_path,
                fetcher=CountingFetcher(fake_closes()))
    got = load_closes("XYZ", "2019-01-01", "2019-12-31", cache_dir=tmp_path,
                      offline=True, fetcher=refuse)
    assert len(got) > 0


def test_cache_not_covering_range_is_refetched_whole(tmp_path):
    load_closes("XYZ", "2019-06-01", "2019-12-31", cache_dir=tmp_path,
                fetcher=CountingFetcher(fake_closes("2019-06-01")))
    fetch = CountingFetcher(fake_closes())
    got = load_closes("XYZ", "2019-01-01", "2019-12-31", cache_dir=tmp_path, fetcher=fetch)
    assert fetch.calls == 1
    assert got.index.min() == pd.Timestamp("2019-01-01")


def test_offline_miss_is_an_error(tmp_path):
    with pytest.raises(RuntimeError, match="offline"):
        load_closes("XYZ", "2019-01-01", "2019-12-31", cache_dir=tmp_path,
                    offline=True, fetcher=refuse)


def test_offline_partial_cache_is_an_error(tmp_path):
    load_closes("XYZ", "2019-06-01", "2019-12-31", cache_dir=tmp_path,
                fetcher=CountingFetcher(fake_closes("2019-06-01")))
    with pytest.raises(RuntimeError, match="offline"):
        load_closes("XYZ", "2019-01-01", "2019-12-31", cache_dir=tmp_path,
                    offline=True, fetcher=refuse)


def test_corrupt_cache_is_rejected(tmp_path):
    path = cache_path("XYZ", tmp_path)
    path.write_text("date,close\n2019-01-02,100\n2019-01-02,101\n")
    with pytest.raises(ValueError, match="corrupt"):
        load_closes("XYZ", "2019-01-02", "2019-01-03", cache_dir=tmp_path, fetcher=refuse)
