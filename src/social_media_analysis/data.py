"""Dataset loading and schema normalization."""
from pathlib import Path
import pandas as pd


def _standardize_tweets(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Normalize common tweet/user column spellings to Tweet and User."""
    frame = frame.copy()
    lower_map = {str(c).strip().lower(): c for c in frame.columns}
    tweet_col = next((lower_map[k] for k in ("tweet", "text", "content", "full_text") if k in lower_map), None)
    user_col = next((lower_map[k] for k in ("user", "username", "screen_name", "author", "user_name") if k in lower_map), None)
    if tweet_col is None:
        raise ValueError(f"{source}: could not find a tweet-text column. Available columns: {list(frame.columns)}")
    if user_col is None:
        raise ValueError(f"{source}: could not find a user/author column. Available columns: {list(frame.columns)}")
    frame = frame.rename(columns={tweet_col: "Tweet", user_col: "User"})
    frame["Tweet"] = frame["Tweet"].fillna("").astype(str)
    frame["User"] = frame["User"].fillna("").astype(str).str.strip()
    return frame[(frame["Tweet"].str.strip() != "") & (frame["User"] != "")].copy()


def load_tweet_sources(data_dir: str | Path) -> pd.DataFrame:
    """Load the three political tweet CSVs referenced by the Colab notebook."""
    data_dir = Path(data_dir)
    names = ["tweets.csv", "BJP_tweets.csv", "modi_reviews.csv", "rahul_reviews.csv"]
    frames = []
    for name in names:
        path = data_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. See README.md for dataset setup.")
        frames.append(_standardize_tweets(pd.read_csv(path), name))
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["User", "Tweet"])


def load_sentiment140(path: str | Path) -> pd.DataFrame:
    """Load the classic 6-column Sentiment140 CSV (headerless, latin-1)."""
    frame = pd.read_csv(path, encoding="latin-1", header=None)
    if frame.shape[1] < 6:
        raise ValueError("Sentiment140 file should contain at least 6 columns.")
    frame = frame[[0, 5]].copy()
    frame.columns = ["sentiment", "tweet"]
    # Original Sentiment140 labels are 0 (negative) and 4 (positive).
    frame["sentiment"] = frame["sentiment"].map({0: 0, 4: 1})
    frame = frame.dropna(subset=["sentiment", "tweet"])
    frame["sentiment"] = frame["sentiment"].astype(int)
    return frame
