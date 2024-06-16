from trafilatura import extract, fetch_url as trafilatura_fetch_url
from typing import Iterator, Optional, Sequence
import requests

from llm_agents_introduction.alpha_vantage import NewsLink


def fetch_url(url: str) -> Optional[str]:
    try:
        response = fetch_url(url)
        if response.status_code == 200:
            return response.text
        else:
            print(f'Download return status {response.status_code}: {url}')
    except requests.exceptions.RequestException:
        print(f"Failed to download: {url}")

    return None


def extract_text_from_url(url: str) -> Optional[str]:
    print(f"Downloading: {url}")
    downloaded = trafilatura_fetch_url(url)

    if not downloaded:
        return None

    print(f"Extracting: {url}")
    text = extract(downloaded)

    return text


def news_links_with_text(
    news_links: Sequence[NewsLink],
    min_words: int = 100,
    max_words: int = 1000,
) -> Iterator[tuple[NewsLink, str]]:
    """
    trafilatura respects the robot.txt file of sites so won't be able to fetch the text of all sites.
    this returns a iterator of just the items that were successfully fetched with their text.

    We also discard articles if they have less than so many words as they are likely not useful.
    Typically this is where they're just a video or a link to something else.
    """
    for link in news_links:
        text = extract_text_from_url(link.url)

        if not text:
            continue

        words = word_count(text)

        if words >= min_words and words <= max_words:
            yield (link, text)


def word_count(text: str) -> int:
    """
    Very lame word count.
    """
    return len(text.split())
