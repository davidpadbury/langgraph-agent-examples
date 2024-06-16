from typing import TypedDict
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from llm_agents_introduction.models import gpt_4o

_system_prompt_text = """
You are a finance writer that creates punchy succinct descriptions of a bunch of news article summaries to annotate graphs.

Consider provided summaries and then produce a punchy description of them.
""".strip()


_human_prompt_text = """
Here are summaries of articles about {company_name} describing why their stock price is exhibiting the follow trend: {trend}.

Create a punchy description of the summaries that is no more than a few sentences long.

{summaries}
""".strip()


class SummariesSummarizerInput(TypedDict):
    company_name: str
    trend: str
    title_with_points: list[tuple[str, list[str]]]


def _format_input(input_data: SummariesSummarizerInput) -> dict:
    result = {
        "company_name": input_data["company_name"],
        "trend": input_data["trend"],
        "summaries": "\n\n".join(
            [
                f"{title}{'\n - '.join(points)}"
                for title, points in input_data["title_with_points"]
            ]
        ),
    }

    return result


def create_summaries_summarizer() -> Runnable[SummariesSummarizerInput, str]:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _system_prompt_text),
            ("human", _human_prompt_text),
        ]
    )

    return RunnableLambda(func=_format_input) | prompt | gpt_4o | StrOutputParser()
