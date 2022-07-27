import re
import string
from collections import Counter
from typing import Optional, List

import string
import re

from nltk.corpus import words
from nltk.corpus import wordnet as wn

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
)


def extract_acronyms(text: str, nltk_words: Optional[List] = None, min_length: int = 3):

    if nltk_words is None:
        nltk_words = set(words.words())

    # Extract the cards that are all caps
    pattern = "[A-Z]+"
    capitals_words = [
        w.strip() for w in re.findall(pattern, text) if len(w) >= min_length
    ]

    count = Counter(capitals_words)

    # Remove words efficienctly in a set lookup
    non_words = [w for w in set(capitals_words) if w.lower() not in nltk_words]

    # Do a more sufisticated filter on smaller subset.
    filtered_non_words = [w for w in non_words if len(wn.synsets(w)) == 0]

    # Return the Acronyms and Counts
    return {k: count[k] for k in filtered_non_words}


def preprocess(text: str):
    # strip out punctuation and numbers
    punctuation = string.punctuation + "0123456789"
    replacement = ""

    # Create a punctuation dictionary which allows for blank
    # replacements
    replacement_dict = {a: replacement for a in punctuation}

    # remove new lines
    replacement_dict["\n"] = " "

    s = text.translate(str.maketrans(replacement_dict))
    s = s.strip(" ")
    s = s.lower()
    return re.sub(r"\s\s+", " ", s)


class Preprocessor(BaseEstimator, TransformerMixin):
    """
    This transformer performs basic string pre-processing.
    """

    def __init__(self):
        """Dummy Init for demo."""
        pass

    def fit(self, X, Y=None):
        """
        This transformer does not require to be fitted
        """
        return self

    def transform(self, X, Y=None):
        """
        Preprocess the input strings

        Parameters
        -----------
        X : list
        list of strings to transform

        Y : list
        list of target labels (this is not a fitted model, hence ignored)
        """

        return [preprocess(text) for text in X]

    def preprocess(self, text):

        # remove punctuation and numericals
        punctuation = string.punctuation + "0123456789"
        replacement = ""
        # Create a punctuation dictionary which allows for blank
        # replacements
        replacement_dict = {a: replacement for a in punctuation}
        replacement_dict["\n"] = " "
        s = text.translate(str.maketrans(replacement_dict))

        s = s.strip(" ")
        # lower the string for better word collapsing
        s = s.lower()
        # Remove any large ammounts of spaces
        return re.sub(r"\s\s+", " ", s)


def print_metrics(title, y_pred, y_true):
    print(title)
    print(confusion_matrix(y_pred, y_true))
    print("Accuracy:", accuracy_score(y_pred, y_true))
    print("Precision:", precision_score(y_pred, y_true))
    print("Recall:", recall_score(y_pred, y_true))
