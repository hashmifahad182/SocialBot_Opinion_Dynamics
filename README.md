# Social Media Sentiment and Network Influence Analysis

A Python project based on a Google Colab academic notebook. It combines tweet sentiment classification, heuristic user-authenticity scoring, mention-network construction, and PageRank-based influence analysis.

> **Scope note:** The uploaded notebook does not implement a supervised bot-vs-human classifier or the disruptive/propaganda opinion-dynamics simulation shown in other project screenshots. This repository documents the functionality present in the uploaded notebook. Add those components only if they are part of the final, tested codebase.

## What it does

1. Cleans tweet text and trains a TF-IDF + Logistic Regression sentiment classifier on Sentiment140.
2. Predicts a positive-sentiment probability for political tweets and maps it to Positive, Neutral, or Negative using thresholds of 0.6 and 0.4.
3. Computes a **heuristic authenticity score** from vocabulary richness, variation in predicted sentiment, and tweet count. This is a project-specific heuristic, not a validated probability that an account is genuine.
4. Builds a directed interaction graph from `@mentions` (tweet author → mentioned handle).
5. Computes PageRank and combines it with the heuristic authenticity score to rank users.
6. Produces CSV summaries and visualizations.

## My contribution (adapt to your actual role)

For the network-analysis component:
- Constructed a directed mention graph using NetworkX.
- Used PageRank to estimate structural influence in the interaction network.
- Combined network influence with a heuristic user-authenticity measure.
- Visualized top-ranked accounts and cumulative PageRank distribution.

This is a collaborative project. Keep only bullets that accurately describe your own work.

## Repository structure

```text
.
├── notebooks/
│   └── social_media_sentiment_network_analysis.ipynb
├── src/social_media_analysis/
│   ├── data.py
│   ├── text_processing.py
│   ├── sentiment.py
│   ├── network_analysis.py
│   └── pipeline.py
├── data/
│   ├── raw/          # dataset files go here; ignored by Git
│   └── processed/
├── results/          # generated output; ignored by Git
├── run_analysis.py
├── requirements.txt
└── README.md
```

## Dataset setup

The original notebook references these files:

- `tweets.csv`
- `BJP_tweets.csv`
- `modi_reviews.csv`
- `rahul_reviews.csv`
- `training.1600000.processed.noemoticon.csv` (Sentiment140)

Place files in `data/raw/`. The loader expects a tweet-text column such as `Tweet`, `tweet`, `text`, `content`, or `full_text`, and a user column such as `User`, `user`, `username`, `screen_name`, `author`, or `user_name`. Check each dataset's license and terms before sharing or redistributing it. Dataset column schemas may need adjustment to match the source files.

The original notebook used Kaggle dataset identifiers:

- `noorsaeed/indian-election-sentiment-dataset-tweets`
- `adritpal08/dataset-of-indian-politics-tweets-and-reactions`
- `anveshrao123/indian-political-tweets`
- `kazanova/sentiment140`

Use Kaggle credentials privately if required. **Never upload `kaggle.json`, API tokens, or private data to GitHub.**

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Activate it:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the modular pipeline from the repository root:

```bash
python run_analysis.py --data-dir data/raw --output-dir results
```

If the Sentiment140 file is stored elsewhere:

```bash
python run_analysis.py --data-dir data/raw --sentiment140 path/to/training.1600000.processed.noemoticon.csv --output-dir results
```

The datasets are not included in this repository. First execution may take time because Sentiment140 contains many tweets and the classifier must be trained.

## Outputs

The pipeline writes the following files to `results/`:

- `tweet_analysis.csv`
- `user_authenticity.csv`
- `user_influence.csv`
- `top_10_influential_users.csv`
- `target_group_sentiment_counts.csv`
- `sentiment_model_metrics.txt`
- `top_10_influence.png` (when results are available)
- `target_group_sentiment.png` (when results are available)

## Method and limitations

- The classifier is trained on Sentiment140's binary labels (0 = negative, 4 = positive, mapped internally to 0/1).
- Sentiment labels use a simple probability threshold: below 0.4 is Negative, above 0.6 is Positive, and values between the thresholds are Neutral.
- Target-group assignment is keyword-based (`Modi/BJP` and `Rahul/Congress`), so it can misclassify tweets that mention terms in another context.
- PageRank is a graph-structure metric; it is not proof of real-world influence.
- The authenticity score is an experimental heuristic, not a calibrated confidence score or proof that an account is human.
- Mentions can point to accounts absent from the source dataset. Such handles may appear as graph nodes but will not have an authenticity score.
- The result is sensitive to dataset quality, duplicate tweets, missing user identifiers, preprocessing, and model assumptions.

## Future work

- Validate the authenticity heuristic against labeled human/bot accounts.
- Add temporal network features and coordinated-retweet analysis.
- Integrate a tested bot-classification model and opinion-dynamics simulations if those modules are available in the final project.
- Evaluate sentiment and influence results with stronger baselines and documented metrics.

## Academic use

This repository is a cleaned and modularized presentation of an academic project. Update the authors/team section, data-source citations, and personal contribution before publishing. Respect dataset licenses and privacy requirements.
