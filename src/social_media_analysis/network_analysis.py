"""Mention graph construction and PageRank-based influence scoring."""
import networkx as nx
import pandas as pd
from .text_processing import extract_mentions


def build_mention_graph(tweets: pd.DataFrame) -> nx.DiGraph:
    """Build a directed user-to-mentioned-account graph from Tweet/User columns."""
    graph = nx.DiGraph()
    graph.add_nodes_from(tweets["User"].dropna().astype(str).unique())
    for row in tweets[["User", "Tweet"]].itertuples(index=False):
        source = str(row.User).strip()
        for target in extract_mentions(row.Tweet):
            if source and target:
                graph.add_edge(source, target)
    return graph


def calculate_influence(graph: nx.DiGraph, authenticity: pd.DataFrame) -> pd.DataFrame:
    """Join PageRank with the notebook's heuristic authenticity score."""
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=["user", "pagerank", "authenticity", "final_score"])
    pagerank = nx.pagerank(graph)
    rank_df = pd.DataFrame({"user": list(pagerank.keys()), "pagerank": list(pagerank.values())})
    result = rank_df.merge(authenticity[["user", "authenticity", "tweet_count"]], on="user", how="left")
    result["authenticity"] = result["authenticity"].fillna(0.0)
    result["final_score"] = result["pagerank"] * result["authenticity"]
    return result.sort_values("final_score", ascending=False).reset_index(drop=True)
