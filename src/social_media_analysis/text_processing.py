"""Text cleaning and tweet-level feature helpers."""
import re


def clean_text(text: object) -> str:
    """Normalize tweet text using the rules used in the original notebook."""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()


def vocabulary_richness(text: object) -> float:
    """Return unique-token ratio; a small denominator guard handles empty text."""
    words = str(text).split()
    return len(set(words)) / (len(words) + 1)


def extract_mentions(text: object) -> list[str]:
    """Extract @handles from the original, uncleaned tweet text."""
    return re.findall(r"@([A-Za-z0-9_]+)", str(text))
