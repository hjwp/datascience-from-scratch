import math
from typing import Dict, List, Set
from dataclasses import dataclass
from collections import defaultdict
from bayes import TrainingMessage, NaiveBayesClassifier
import pytest


def test_naive_bayes_classifier_counts():
    messages = [
        TrainingMessage("spam rules", is_spam=True),
        TrainingMessage("ham rules", is_spam=False),
        TrainingMessage("hello ham", is_spam=False),
    ]

    model = NaiveBayesClassifier(k=0.5)
    model.train(messages)

    print(model.tokens)
    assert model.tokens == {"spam", "ham", "rules", "hello"}
    assert model.total_spam_messages == 1
    assert model.total_ham_messages == 2
    assert model.token_spam_counts == {"spam": 1, "rules": 1}
    assert model.token_ham_counts == {"ham": 2, "rules": 1, "hello": 1}


def test_naive_bayes_token_conditional_probabilites():
    messages = [
        TrainingMessage("spam rules", is_spam=True),
        TrainingMessage("ham rules", is_spam=False),
        TrainingMessage("hello ham", is_spam=False),
    ]

    k = 0.5
    model = NaiveBayesClassifier(k=k)
    model.train(messages)

    assert model.p_token_given_spam("spam") == (1 + k) / (1 + 2 * k)
    assert model.p_token_given_spam("rules") == (1 + k) / (1 + 2 * k)
    assert model.p_token_given_spam("ham") == (0 + k) / (1 + 2 * k)
    assert model.p_token_given_ham("spam") == (0 + k) / (2 + 2 * k)
    assert model.p_token_given_ham("rules") == (1 + k) / (2 + 2 * k)
    assert model.p_token_given_ham("ham") == (2 + k) / (2 + 2 * k)


def test_naive_bayes_predictions():
    messages = [
        TrainingMessage("spam rules", is_spam=True),
        TrainingMessage("ham rules", is_spam=False),
        TrainingMessage("hello ham", is_spam=False),
    ]

    k = 0.5
    model = NaiveBayesClassifier(k=k)
    model.train(messages)

    text = "hello spam"

    probs_if_spam = [
        (1 + 0.5) / (1 + 2 * 0.5),      # "spam"  (present)
        1 - (0 + 0.5) / (1 + 2 * 0.5),  # "ham"   (not present)
        1 - (1 + 0.5) / (1 + 2 * 0.5),  # "rules" (not present)
        (0 + 0.5) / (1 + 2 * 0.5)       # "hello" (present)
    ]
    
    probs_if_ham = [
        (0 + 0.5) / (2 + 2 * 0.5),      # "spam"  (present)
        1 - (2 + 0.5) / (2 + 2 * 0.5),  # "ham"   (not present)
        1 - (1 + 0.5) / (2 + 2 * 0.5),  # "rules" (not present)
        (1 + 0.5) / (2 + 2 * 0.5),      # "hello" (present)
    ]

    p_if_spam = math.exp(sum(math.log(p) for p in probs_if_spam))
    p_if_ham = math.exp(sum(math.log(p) for p in probs_if_ham))
    # Should be about 0.83
    assert model.predict(text) == pytest.approx(p_if_spam / (p_if_spam + p_if_ham), 1e5)
