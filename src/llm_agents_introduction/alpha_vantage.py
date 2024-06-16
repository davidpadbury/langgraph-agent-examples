from typing import Optional, Protocol
import requests
import os
import pandas as pd
from dataclasses import dataclass, asdict
from functools import lru_cache
from datetime import date, datetime, time, timedelta


class HasDate(Protocol):
    date: date


@dataclass
class SearchResult:
    name: str
    symbol: str
    type: str
    region: str
    currency: str


@dataclass
class TimeSeriesDaily:
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class TimeSeriesDailyAdjusted:
    date: date
    open: float
    high: float
    low: float
    close: float
    adjusted_close: float
    volume: int
    dividend_amount: float
    split_coefficient: float


@dataclass
class Overview:
    symbol: str
    name: str
    description: str
    market_cap: float
    shares_outstanding: float


@dataclass
class TickerSentiment:
    ticker: str
    relevance_score: float
    ticker_sentiment_score: float


@dataclass
class NewsLink:
    title: str
    url: str
    summary: str
    overall_sentiment_score: float
    published_on: date
    ticker_sentiment: list[TickerSentiment]

    def sentiment_for_ticker(self, ticker: str) -> Optional[TickerSentiment]:
        return next(
            filter(lambda item: item.ticker == ticker, self.ticker_sentiment), None
        )


@dataclass
class Split:
    effective_date: date
    factor: float


@dataclass
class MarketCapDaily:
    date: date
    market_cap: float


def _parse_date(date_text: str) -> date:
    return datetime.strptime(date_text, "%Y-%m-%d").date()


def _format_date(date: datetime) -> str:
    return datetime.combine(date, time(0, 0)).strftime("%Y%m%dT%H%M")


class AlphaVantageClient:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key is required")

        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

    def search(self, term: str) -> list[SearchResult]:
        params = {
            "function": "SYMBOL_SEARCH",
            "keywords": term,
            "datatype": "json",
            "apikey": self.api_key,
        }
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        response_data = response.json()

        def build_search_result(data: dict) -> SearchResult:
            return SearchResult(
                name=data["2. name"],
                symbol=data["1. symbol"],
                type=data["3. type"],
                region=data["4. region"],
                currency=data["8. currency"],
            )

        if "bestMatches" not in response_data:
            raise Exception(f"Unexpected response: {response_data}")

        return list(map(build_search_result, response_data["bestMatches"]))

    def fetch_daily_adjusted(self, symbol: str) -> list[TimeSeriesDailyAdjusted]:
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full",
            "datatype": "json",
            "apikey": self.api_key,
        }
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        response_data = response.json()

        def build_time_series_daily_adjusted(
            date: str, data: dict
        ) -> TimeSeriesDailyAdjusted:
            return TimeSeriesDailyAdjusted(
                date=_parse_date(date),
                open=float(data["1. open"]),
                high=float(data["2. high"]),
                low=float(data["3. low"]),
                close=float(data["4. close"]),
                adjusted_close=float(data["5. adjusted close"]),
                volume=int(data["6. volume"]),
                dividend_amount=float(data["7. dividend amount"]),
                split_coefficient=float(data["8. split coefficient"]),
            )

        time_series_key = "Time Series (Daily)"
        data: dict[str, dict[str, str]] = response_data.get(time_series_key, None)

        if not data:
            raise ValueError(f"Daily prices for not found for symbol: {symbol}")

        return list(
            map(
                lambda item: build_time_series_daily_adjusted(item[0], item[1]),
                data.items(),
            )
        )

    def fetch_daily(self, symbol: str) -> list[TimeSeriesDaily]:
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": "full",
            "datatype": "json",
            "apikey": self.api_key,
        }
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        response_data = response.json()

        def build_time_series_daily(date: str, data: dict) -> TimeSeriesDaily:
            return TimeSeriesDaily(
                date=_parse_date(date),
                open=float(data["1. open"]),
                high=float(data["2. high"]),
                low=float(data["3. low"]),
                close=float(data["4. close"]),
                volume=int(data["5. volume"]),
            )

        time_series_key = "Time Series (Daily)"
        data: dict[str, dict[str, str]] = response_data.get(time_series_key, None)

        if not data:
            raise ValueError(f"Daily prices for not found for symbol: {symbol}")

        return list(
            map(lambda item: build_time_series_daily(item[0], item[1]), data.items())
        )

    def fetch_overview_raw(self, symbol: str) -> dict:
        params = {
            "function": "OVERVIEW",
            "symbol": symbol,
            "datatype": "json",
            "apikey": self.api_key,
        }
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        return response.json()

    def fetch_overiew(self, symbol: str) -> Overview:
        response_data = self.fetch_overview_raw(symbol)

        return Overview(
            symbol=response_data["Symbol"],
            name=response_data["Name"],
            description=response_data["Description"],
            market_cap=float(response_data["MarketCapitalization"]),
            shares_outstanding=float(response_data["SharesOutstanding"]),
        )

    def news_sentiment(
        self, ticker: str, from_date: date, to_date: date
    ) -> list[NewsLink]:
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "time_from": _format_date(from_date),
            "time_to": _format_date(to_date),
            "sort": "RELEVANCE",
            "limit": 1000,
            "apikey": self.api_key,
        }
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        response_data = response.json()

        def build_ticker_sentiment(data: dict) -> TickerSentiment:
            return TickerSentiment(
                ticker=data["ticker"],
                relevance_score=float(data["relevance_score"]),
                ticker_sentiment_score=float(data["ticker_sentiment_score"]),
            )

        def build_news_link(data: dict) -> NewsLink:
            return NewsLink(
                title=data["title"],
                url=data["url"],
                summary=data["summary"],
                published_on=datetime.strptime(data["time_published"], '%Y%m%dT%H%M%S').date(),
                overall_sentiment_score=float(data["overall_sentiment_score"]),
                ticker_sentiment=[
                    build_ticker_sentiment(item) for item in data["ticker_sentiment"]
                ],
            )

        if "feed" not in response_data:
            return []

        return [build_news_link(item) for item in response_data["feed"]]

    def splits(self, symbol: str) -> list[Split]:
        params = {"function": "SPLITS", "symbol": symbol, "apikey": self.api_key}
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        response_data = response.json()

        def build_split(data: dict) -> Split:
            return Split(
                effective_date=_parse_date(data["effective_date"]),
                factor=float(data["split_factor"]),
            )

        return [build_split(item) for item in response_data.get("data", [])]


class AlphaVantageService:
    """
    Wrapper on top of the client to provide caching and other features
    """

    @staticmethod
    def create() -> "AlphaVantageService":
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

        if not api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY env variable is required")

        return AlphaVantageService(AlphaVantageClient(api_key))

    def __init__(self, client: AlphaVantageClient):
        self.client = client

    @lru_cache(maxsize=1024)
    def search(self, term: str) -> list[SearchResult]:
        """Filters to just US equities."""

        def is_us_equity(result: SearchResult) -> bool:
            return result.type == "Equity" and result.region == "United States"

        return list(filter(is_us_equity, self.client.search(term)))

    @lru_cache(maxsize=128)
    def fetch_daily(self, symbol: str) -> list[TimeSeriesDaily]:
        return self.client.fetch_daily(symbol)

    @lru_cache(maxsize=128)
    def fetch_daily_adjusted(self, symbol: str) -> list[TimeSeriesDailyAdjusted]:
        return self.client.fetch_daily_adjusted(symbol)

    @lru_cache(maxsize=1024)
    def overview(self, symbol: str) -> Overview:
        return self.client.fetch_overiew(symbol)

    @lru_cache(maxsize=1024)
    def company_info(self, symbol: str) -> dict:
        return self.client.fetch_overview_raw(symbol)

    def latest_price(self, symbol: str) -> float:
        daily = self.fetch_daily(symbol)

        if len(daily) == 0:
            raise ValueError(f"No data found for {symbol}")

        return daily[0].close

    def _find_daily_for_date(self, symbol: str, date: str) -> TimeSeriesDailyAdjusted:
        daily = self.fetch_daily_adjusted(symbol)
        looking_for_date = _parse_date(date)

        for item in daily:
            # assume sorted in reverse chronological order
            if item.date <= looking_for_date:
                return item

        raise ValueError(f"No data found for {symbol} on {date}")

    def price_on_date(self, symbol: str, date: str) -> float:
        return self._find_daily_for_date(symbol, date).adjusted_close

    def latest_market_cap(self, symbol: str) -> float:
        overview = self.overview(symbol)

        return overview.market_cap

    def market_cap_on_date(self, symbol: str, date: str) -> float:
        market_caps = self.fetch_daily_market_cap(symbol)
        looking_for_date = _parse_date(date)

        for item in market_caps:
            # assume sorted in reverse chronological order
            if item.date <= looking_for_date:
                return item.market_cap

        raise ValueError(f"No data found for {symbol} on {date}")

    @lru_cache(maxsize=1024)
    def fetch_daily_market_cap(self, symbol: str) -> list[MarketCapDaily]:
        overview = self.overview(symbol)  # get the current outstanding shares
        splits = self.splits(symbol)  # get historic splits

        daily = self.fetch_daily_adjusted(symbol)
        df = pd.DataFrame([asdict(price) for price in daily])

        # start with today's outstanding shares
        df["outstanding_shares"] = overview.shares_outstanding

        # apply splits to the value of today's outstanding_shares in reverse order
        for split in sorted(splits, key=lambda x: x.effective_date, reverse=True):
            df.loc[df["date"] < split.effective_date, "outstanding_shares"] /= (
                split.factor
            )

        # calculate market cap on these values
        df["market_cap"] = df["close"] * df["outstanding_shares"]

        # use a moving average to smooth out the market cap
        df["market_cap_ma"] = df["market_cap"].rolling(window=50).mean()

        return [
            MarketCapDaily(date=row["date"], market_cap=row["market_cap"])
            for _, row in df.iterrows()
        ]

    @lru_cache(maxsize=1024)
    def splits(self, symbol: str) -> list[Split]:
        return self.client.splits(symbol)

    @lru_cache(maxsize=1024)
    def fetch_relevant_news(
        self,
        symbol: str,
        from_date: date,
        to_date: date,
    ) -> list[NewsLink]:
        """
        The alpha vantage API returns all news items that may only just mention the ticker
        and may not be mainly about the ticker. So filter the returned this to just items
        focused on the ticker.

        We'll also automatically expand the time range to get more news items if none are returned.
        """

        def item_relevance_score(item: NewsLink) -> float:
            sentiment = item.sentiment_for_ticker(symbol)
            return sentiment.relevance_score if sentiment else 0

        attempt = 0

        while attempt < 5:
            news = self.client.news_sentiment(symbol, from_date, to_date)
            attempt += 1

            # expand the time range to get more news items if none are returned
            if news:
                continue
            else:
                delta = to_date - from_date
                # add 25% of the delta to the range
                expand_range_days = delta.days // 4

                from_date -= timedelta(days=expand_range_days)
                to_date = min(to_date + timedelta(days=expand_range_days), date.today())

        return sorted(news, key=item_relevance_score, reverse=True)
