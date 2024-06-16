from datetime import date
from typing import TypedDict
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableLambda, Runnable
from langchain_core.output_parsers import JsonOutputParser
from llm_agents_introduction.models import gpt_4o
from llm_agents_introduction.aws import upload_file_to_s3

_system_prompt_text = """
You are a stock price chart reviewer.
You must review the stock price chart and identify trends from the chart.

You must respond with a list of at most the 3 most interesting date ranges where some interesting trend seems to be occurring.
Don't feel compelled to find 3 date ranges if there are fewer interesting trends.

Format your result as a json list where each item is a object with the keys: 'start_date', 'end_date', 'notes'.
'notes' can contain a description of what a researcher should look for in the date range.

Example:
```json
[
    {{
        "start_date": "2020-01-01",
        "end_date": "2020-01-31",
        "notes": "Stock price is increasing"
    }},
    {{
        "start_date": "2020-02-01",
        "end_date": "2020-02-28",
        "notes": "Stock price is decreasing"
    }}
]
```
""".strip()


_human_prompt_text = """
Please review the attached chart and return a list of interesting date ranges where something interesting is happening.

This is a chart of the {symbol} stock price from {from_date} to {to_date}.
""".strip()


class StockPriceChartReviewerInput(TypedDict):
    from_date: date
    to_date: date
    symbol: str
    image_path: str


class StockPriceChartReviewerItem(TypedDict):
    start_date: date
    end_date: date
    notes: str

    @staticmethod
    def from_json_dict(data: dict) -> "StockPriceChartReviewerItem":
        return StockPriceChartReviewerItem(
            start_date=date.fromisoformat(data["start_date"]),
            end_date=date.fromisoformat(data["end_date"]),
            notes=data["notes"],
        )


def _format_prompt(input: StockPriceChartReviewerInput) -> dict:
    return {
        "chart_url": upload_file_to_s3(input["image_path"]),
        "detail": f'Chart of the {input['symbol']} stock price from {input['from_date']} to {input['to_date']}',
        "symbol": input["symbol"],
        "from_date": input["from_date"],
        "to_date": input["to_date"],
    }


def create_stock_price_chart_reviewer() -> (
    Runnable[StockPriceChartReviewerInput, list[StockPriceChartReviewerItem]]
):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _system_prompt_text),
            HumanMessagePromptTemplate.from_template(
                [_human_prompt_text, {"image_url": "{chart_url}"}]
            ),
        ]
    )

    return (
        RunnableLambda(func=_format_prompt)
        | prompt
        | gpt_4o
        | JsonOutputParser()
        | RunnableLambda(
            func=lambda x: [
                StockPriceChartReviewerItem.from_json_dict(item) for item in x
            ]
        )
    )
