from typing import Optional, TypedDict
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from llm_agents_introduction.models import gpt_4o

_system_prompt_text = """
You are a finance analyst. You aid a team of researchers in identifying trends in stock price charts.
You review the content of news articles and summarize the key points that could be relevant to a trend in a stock price.

You must respond with a bullet point list of key points from the news article.
Only include the key points that are relevant to the stock price trend. Include no more than 3 main points.
If there are no relevant points in the article, instead respond with "No relevant points found".
""".strip()


_human_prompt_text = """
I am trying to identify why {company_name} during a certain period is showing this trend: {trend}.

The below is a news article with the title: {title}.
Please summarize the key points from the article that could be relevant to the trend in the stock price.

Article:

{article_content}
""".strip()


class NewsArticleSummarizerInput(TypedDict):
    company_name: str
    trend: str
    title: str
    article_content: str


def _trim_bullet_point(text: str) -> str:
    return text.strip().lstrip("- ")


def _format_output(message: str) -> Optional[list[str]]:
    if "No relevant points found" in message:
        return None

    return [_trim_bullet_point(item) for item in message.split("\n")]


def create_news_article_summarizer() -> Runnable[NewsArticleSummarizerInput, str]:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _system_prompt_text),
            ("human", _human_prompt_text),
        ]
    )

    return prompt | gpt_4o | StrOutputParser() | RunnableLambda(func=_format_output)
