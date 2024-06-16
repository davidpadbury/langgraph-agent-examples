from langchain_core.embeddings import Embeddings
from scipy.cluster.hierarchy import linkage, fcluster
from llm_agents_introduction.alpha_vantage import NewsLink
from itertools import groupby

def cluster_news_links(news_links: list[NewsLink], embedding_model: Embeddings) -> list[list[NewsLink]]:
    """
    Use hierarchical clusterings to group news links by their similarity.
    """
    summaries = [news_link.summary for news_link in news_links]
    embeddings = embedding_model.embed_documents(summaries)

    # Compute the linkage matrix
    Z = linkage(embeddings, method='ward')

    # Define a threshold to cut the dendrogram
    threshold = 1.5  # picked this based on some very rough experimentation

    # Form clusters
    clusters = fcluster(Z, threshold, criterion='distance')

    items_with_clusters = list(zip(news_links, clusters))

    grouped_by_cluster = groupby(sorted(items_with_clusters, key=lambda x: x[1]), key=lambda x: x[1])

    grouped_by_cluster = [ [item[0] for item in group] for _, group in grouped_by_cluster ]

    return sorted(grouped_by_cluster, key=lambda x: len(x), reverse=True)