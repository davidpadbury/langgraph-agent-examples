from datetime import date
from dataclasses import asdict
from uuid import uuid4
from langchain_core.tools import BaseTool, tool
from langchain_core.pydantic_v1 import BaseModel, Field
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from llm_agents_introduction.alpha_vantage import AlphaVantageService
from llm_agents_introduction.types import URL
from llm_agents_introduction.aws import upload_to_s3


def generate_daily_chart(
    alpha_vantage: AlphaVantageService,
    symbol: str,
    from_date: date,
    to_date: date,
) -> go.Figure:
    """Generates a chart image of the daily stock prices for the given symbol."""
    daily_prices_raw = alpha_vantage.fetch_daily_adjusted(symbol)
    daily_prices_df = pd.DataFrame([asdict(price) for price in daily_prices_raw])
    selected_prices = daily_prices_df[
        (daily_prices_df["date"] >= from_date) & (daily_prices_df["date"] <= to_date)
    ]
    chart = px.line(
        selected_prices,
        x="date",
        y="adjusted_close",
        title=f"{symbol} share price",
        labels={"date": "Date", "adjusted_close": "Price (USD)"},
    )

    return chart


def generate_market_cap_chart(
    alpha_vantage: AlphaVantageService,
    symbol: str,
    from_date: date,
    to_date: date,
) -> go.Figure:
    """Generates a chart image of the daily stock prices for the given symbol."""
    daily_market_cap = alpha_vantage.fetch_daily_market_cap(symbol)
    df = pd.DataFrame([asdict(item) for item in daily_market_cap])

    selected_prices = df[(df["date"] >= from_date) & (df["date"] <= to_date)]
    chart = px.line(
        selected_prices,
        x="date",
        y="market_cap",
        title=f"{symbol} Market Capitalization",
        labels={"date": "Date", "market_cap": "Market Cap (USD)"},
    )

    return chart


def upload_chart(chart: go.Figure) -> URL:
    """Uploads the given chart to S3 and returns the URL."""
    data = chart.to_image(format="png")
    filename = f"{str(uuid4())}.png"

    return upload_to_s3(data, filename)


class GenerateChartInput(BaseModel):
    symbol: str = Field(description="The stock symbol to generate the chart for")
    from_date: str = Field(
        description="Start date of the chart in the format 'YYYY-MM-DD'"
    )
    to_date: str = Field(
        description="Inclusive end date of the chart in the format 'YYYY-MM-DD'"
    )


def create_chart_tools(alpha_vantage: AlphaVantageService) -> list[BaseTool]:
    @tool(args_schema=GenerateChartInput)
    def generate_share_price_chart(
        symbol: str,
        from_date: str,
        to_date: str,
    ) -> str:
        """Generates a chart of the daily share price for the given symbol. Returns the URL to the image."""
        from_date = date.fromisoformat(from_date)
        to_date = date.fromisoformat(to_date)

        chart = generate_daily_chart(alpha_vantage, symbol, from_date, to_date)
        chart_url = upload_chart(chart)

        return chart_url

    @tool(args_schema=GenerateChartInput)
    def generate_market_capitalization_chart(
        symbol: str,
        from_date: str,
        to_date: str,
    ) -> str:
        """Generates a chart of the market capitalization of the symbol over time. Returns the URL to the image."""
        from_date = date.fromisoformat(from_date)
        to_date = date.fromisoformat(to_date)

        chart = generate_market_cap_chart(alpha_vantage, symbol, from_date, to_date)
        chart_url = upload_chart(chart)

        return chart_url

    return [
        generate_share_price_chart,
        generate_market_capitalization_chart,
    ]
