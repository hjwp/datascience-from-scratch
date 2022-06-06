from typing import Dict, List, Set
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class TrainingMessage:
    text: str
    is_spam: bool


class NaiveBayesClassifier:
    def __init__(self, k: float) -> None:
        """
        k is the smoothing factor
        """
        self.k = k
        self.tokens = set()  # type: Set[str]
        self.total_spam_messages = 0
        self.total_ham_messages = 0
        self.token_spam_counts = defaultdict(int)  # type: Dict[str, int]
        self.token_ham_counts = defaultdict(int)  # type: Dict[str, int]

    def train(self, messages: List[TrainingMessage]) -> None:
        for message in messages:
            for token in message.text.split():
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_counts[token] += 1
                else:
                    self.token_ham_counts[token] += 1

            if message.is_spam:
                self.total_spam_messages += 1
            else:
                self.total_ham_messages += 1

    def predict(self, text: str) -> float:
        message_tokens = text.split()

        p_all_tokens_given_spam = 1.0
        p_all_tokens_given_ham = 1.0

        for token in self.tokens:
            if token in message_tokens:
                p_all_tokens_given_spam *= self.p_token_given_spam(token)
                p_all_tokens_given_ham *= self.p_token_given_ham(token)
            else:
                p_all_tokens_given_spam *= (1 - self.p_token_given_spam(token))
                p_all_tokens_given_ham *= (1 - self.p_token_given_ham(token))

        # the return p(spam) / p(spam) + p(ham)
        return p_all_tokens_given_spam / (p_all_tokens_given_spam + p_all_tokens_given_ham)


    def p_token_given_spam(self, token: str) -> float:
        # num times we've seen this token in spam divided by total num spam messages seen
        token_spam_count = self.token_spam_counts[token] + self.k
        return token_spam_count / (self.total_spam_messages + self.k * 2)

    def p_token_given_ham(self, token: str) -> float:
        # num times we've seen this token in ham divided by total num ham messages seen
        token_ham_count = self.token_ham_counts[token] + self.k
        return token_ham_count / (self.total_ham_messages + self.k * 2)
