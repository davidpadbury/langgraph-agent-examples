import json
from langchain_core.tools import BaseTool, tool
from langchain_core.pydantic_v1 import BaseModel, Field
from llm_agents_introduction.alpha_vantage import AlphaVantageService


class SymbolAndDateInput(BaseModel):
    symbol: str = Field(description="The symbol of the company (not the company name)")
    date: str = Field(
        description="The date to lookup a value for in the format 'YYYY-MM-DD'"
    )


def create_market_data_tools(alpha_vantage: AlphaVantageService) -> list[BaseTool]:
    @tool
    def search_for_symbol(company_name: str) -> str:
        """Looks up possible symbols for a given company name."""
        results = alpha_vantage.search(company_name)

        return json.dumps(
            [{"symbol": result.symbol, "name": result.name} for result in results],
            indent=2,
        )

    @tool(args_schema=SymbolAndDateInput)
    def get_market_capitalization_on_date(symbol: str, date: str) -> float:
        """Returns the market capitalization in USD of the given symbol on the given date."""
        return alpha_vantage.market_cap_on_date(symbol, date)

    @tool(args_schema=SymbolAndDateInput)
    def get_share_price_on_date(symbol: str, date: str) -> float:
        """Returns the share price in USD of the given symbol on the given date (this is adjusted for splits)."""
        return alpha_vantage.price_on_date(symbol, date)

    @tool
    def get_company_info(symbol: str) -> str:
        """
        Returns the company information for the given symbol.
        Includes company info such as sector/industry/location, financial ratios, and other key metrics for the equity specified"""
        return json.dumps(alpha_vantage.company_info(symbol), indent=2)

    return [
        search_for_symbol,
        get_market_capitalization_on_date,
        get_share_price_on_date,
        get_company_info,
    ]
