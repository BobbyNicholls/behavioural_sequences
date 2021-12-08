"""
This is intended to demonstrate the principle of tf-idf in the matching of generic sequences of data.

"Piece together digital trails to reconnect with the individual that's going to generate revenue for you.. each action
you take is storing some kind of signal and generating mountains of data"
Jonathan Lakin

Given this:
    "Our clients typically have tens of millions of customers and billions of interaction events each day... We use
    machine learning to predict behavioural indicators from vast heterogeneous datasets", how might we consider mapping
    generic sequences of events to interesting outcomes.

"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_generation import get_random_sequence

SEQUENCES = 500


if __name__ == "__main__":
    behavioural_sequences = [get_random_sequence() for _ in range(SEQUENCES)]
    behavioural_df = pd.DataFrame(
        data=behavioural_sequences, columns=["sequence", "outcome"]
    )
    vectoriser = TfidfVectorizer(analyzer="char")
    X = vectoriser.fit_transform(behavioural_df["sequence"])
    df = pd.DataFrame(X.toarray(), columns=vectoriser.get_feature_names())
    cosine_df = pd.DataFrame(cosine_similarity(df))
    cosine_df["outcome"] = behavioural_df["outcome"]
    outcome_positive_df = cosine_df[cosine_df["outcome"] == 1]
    outcome_positive_pos_comp_df = outcome_positive_df[list(outcome_positive_df.index)]
    outcome_positive_neg_comp_df = outcome_positive_df[
        [col for col in cosine_df.columns if col not in set(outcome_positive_df.index)]
    ]

    outcome_positive_neg_comp_df.mean().mean()
    outcome_positive_pos_comp_df.mean().mean()

    sorted_cosine_df = cosine_df.sort_values(["outcome"], ascending=False)
    print(behavioural_sequences)
