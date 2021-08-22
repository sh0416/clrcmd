import random
from functools import partial
from typing import Dict, Set, Tuple

import numpy as np
from transformers import RobertaTokenizer


def get_pairs(word):
    """Return set of symbol pairs in a word

    :param word: List of symbol
    :type word: List[str]
    :return: Set of pair observed in the word
    :rtype: Set[Tuple[str, str]]
    """
    return set(zip(word, word[1:]))


def get_priority(pair, dropped_ranks, bpe_ranks):
    """Get priority of given pair

    If the pair is dropped or not existed in bpe_ranks, then the priority is infinity

    :param pair: Symbol pair
    :type pair: Tuple[str, str]
    :param dropped_ranks: Set of pair that dropped during tokenization process
    :type dropped_ranks: Set[Tuple[str, str]]
    :param bpe_ranks: Dictionary that stores the priority of each pair
    :type bpe_ranks: Dict[Tuple[str, str], int]
    :return: Priority
    :rtype: float
    """
    return (
        float("inf")
        if pair in dropped_ranks
        else bpe_ranks.get(pair, float("inf"))
    )


class RobertaTokenizerDropout(RobertaTokenizer):
    """Tokenizer with BPE dropout implementation"""

    def bpe(self, token: str) -> str:
        """Byte pair encoding process

        BPE dropout is implemented in this code

        :param token:
        :type token: str
        :return:
        :rtype:
        """
        word = tuple(token)
        pairs = get_pairs(word)
        dropped_ranks = set(
            filter(
                lambda x: np.random.binomial(1, self.bpe_dropout_prob), pairs
            )
        )
        observed_ranks = set(pairs)

        if not pairs:
            return token

        _get_priority = partial(get_priority, bpe_ranks=self.bpe_ranks)

        while True:
            bigram = min(pairs, key=lambda x: _get_priority(x, dropped_ranks))
            if _get_priority(bigram, dropped_ranks) == float("inf"):
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if (
                    word[i] == first
                    and i < len(word) - 1
                    and word[i + 1] == second
                ):
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
                for pair in pairs:
                    if pair not in observed_ranks:
                        if np.random.binomial(1, self.bpe_dropout_prob):
                            dropped_ranks.add(pair)
                        observed_ranks.add(pair)
        word = " ".join(word)
        return word
