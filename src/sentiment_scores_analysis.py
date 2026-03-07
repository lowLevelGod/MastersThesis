import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def set_publication_style():
    sns.set_theme(style="white")  # clean white background
    
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "pdf.fonttype": 42,   # editable text in Illustrator
    })

set_publication_style()

# -----------------------------
# Configuration
# -----------------------------

SENTIMENT_COLS = [
    "finbert_neg",
    "finbert_neu",
    "finbert_pos",
    "finbert_neg_std",
    "finbert_neu_std",
    "finbert_pos_std",
    "finbert_polarity_std"
]

df = pd.read_parquet("us_market_dataset_finbert.parquet")  # or csv

sent_df = df[SENTIMENT_COLS].dropna()

fig, axes = plt.subplots(3, 3, figsize=(14,10))
axes = axes.flatten()

for i, col in enumerate(SENTIMENT_COLS):
    sns.histplot(sent_df[col], bins=50, kde=True, ax=axes[i])
    axes[i].set_title(col)

# hide unused axes
for j in range(len(SENTIMENT_COLS), len(axes)):
    axes[j].axis("off")

plt.suptitle("Distribution of Sentiment Features")
plt.tight_layout()
plt.savefig("sentiment_scores_distributions.pdf")