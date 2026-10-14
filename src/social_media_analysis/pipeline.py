"""End-to-end analysis pipeline for the project's canonical input files."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .data import load_tweet_sources, load_sentiment140
from .text_processing import clean_text, vocabulary_richness
from .sentiment import train_sentiment_model, sentiment_label
from .network_analysis import build_mention_graph, calculate_influence


def run(data_dir: str | Path, output_dir: str | Path, sentiment140_path: str | Path | None = None) -> dict:
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sentiment_path = Path(sentiment140_path) if sentiment140_path else data_dir / "training.1600000.processed.noemoticon.csv"
    train_df = load_sentiment140(sentiment_path)
    sentiment_model = train_sentiment_model(train_df)

    tweets = load_tweet_sources(data_dir)
    tweets["clean"] = tweets["Tweet"].map(clean_text)
    tweets["tweet_len"] = tweets["Tweet"].str.len()
    tweets = tweets[tweets["tweet_len"] >= 20].copy()
    tweets["sentiment_score"] = tweets["Tweet"].map(sentiment_model.positive_probability)
    tweets["sentiment_label"] = tweets["sentiment_score"].map(sentiment_label)
    tweets["vocab"] = tweets["clean"].map(vocabulary_richness)

    authenticity = tweets.groupby("User", as_index=False).agg(
        tweet_count=("Tweet", "count"),
        sentiment_var=("sentiment_score", "std"),
        vocab_score=("vocab", "mean"),
    )
    # Preserve the notebook's heuristic formula, with undefined one-tweet std set to zero.
    authenticity["sentiment_var"] = authenticity["sentiment_var"].fillna(0.0)
    authenticity["authenticity"] = (
        (authenticity["vocab_score"] + authenticity["sentiment_var"]) /
        authenticity["tweet_count"].clip(lower=1)
    )

    text = tweets["clean"]
    tweets["target_group"] = pd.NA
    tweets.loc[text.str.contains(r"\b(modi|bjp|nda)\b", regex=True), "target_group"] = "Modi/BJP"
    tweets.loc[text.str.contains(r"\b(rahul|gandhi|congress|upa)\b", regex=True), "target_group"] = "Rahul/Congress"

    graph = build_mention_graph(tweets)
    influence = calculate_influence(graph, authenticity)
    top10 = influence.head(10)

    tweets.to_csv(output_dir / "tweet_analysis.csv", index=False)
    authenticity.to_csv(output_dir / "user_authenticity.csv", index=False)
    influence.to_csv(output_dir / "user_influence.csv", index=False)
    top10.to_csv(output_dir / "top_10_influential_users.csv", index=False)
    with open(output_dir / "sentiment_model_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Test accuracy: {sentiment_model.test_accuracy:.4f}\n\n")
        f.write(sentiment_model.report)
        f.write("\nConfusion matrix (true rows; predicted columns; negative, positive):\n")
        f.write(str(sentiment_model.confusion))

    if not top10.empty:
        plt.figure(figsize=(9, 5))
        sns.barplot(data=top10, x="user", y="final_score")
        plt.xticks(rotation=70, ha="right")
        plt.ylabel("PageRank × authenticity heuristic")
        plt.title("Top 10 users by combined influence score")
        plt.tight_layout()
        plt.savefig(output_dir / "top_10_influence.png", dpi=160)
        plt.close()

    labeled = tweets[tweets["target_group"].notna()]
    sentiment_counts = labeled.groupby(["target_group", "sentiment_label"]).size().unstack(fill_value=0)
    sentiment_counts.to_csv(output_dir / "target_group_sentiment_counts.csv")
    if not sentiment_counts.empty:
        ax = sentiment_counts.reindex(columns=["Negative", "Neutral", "Positive"], fill_value=0).plot(kind="bar", figsize=(8, 5))
        ax.set_ylabel("Number of tweets")
        ax.set_title("Sentiment by keyword-based target group")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / "target_group_sentiment.png", dpi=160)
        plt.close()

    return {
        "sentiment_test_accuracy": sentiment_model.test_accuracy,
        "tweets_analyzed": int(len(tweets)),
        "graph_nodes": int(graph.number_of_nodes()),
        "graph_edges": int(graph.number_of_edges()),
        "top_users": top10,
    }
