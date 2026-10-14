"""Run the social media sentiment and network influence analysis."""
import argparse
from src.social_media_analysis.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw", help="Directory containing the CSV datasets")
    parser.add_argument("--output-dir", default="results", help="Directory for generated outputs")
    parser.add_argument("--sentiment140", default=None, help="Optional path to Sentiment140 CSV")
    args = parser.parse_args()
    summary = run(args.data_dir, args.output_dir, args.sentiment140)
    print("Analysis completed")
    for key, value in summary.items():
        if key != "top_users":
            print(f"{key}: {value}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
