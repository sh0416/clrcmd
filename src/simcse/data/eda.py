# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou
import random
import re
from random import shuffle
from typing import List

import nltk

nltk.download("punkt")
nltk.download("wordnet")
from nltk.corpus import wordnet
from nltk.tokenize.treebank import TreebankWordDetokenizer

# stop words list
# fmt: off
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
    'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
    'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 
    'should', 'now', '']
# fmt: on


def get_only_chars(line: str) -> str:
    """Cleaning up text"""
    clean_line = ""
    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ")  # replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        clean_line += char if char in "qwertyuiopasdfghjklzxcvbnm " else " "

    clean_line = re.sub(" +", " ", clean_line)  # delete extra spaces
    if clean_line[0] == " ":
        clean_line = clean_line[1:]
    return clean_line


def synonym_replacement(words: List[str], n: int) -> List[str]:
    """Replace n words in the sentence with synonyms from wordnet"""
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:  # only replace up to n words
            break

    # this is stupid but we need it, trust me
    sentence = " ".join(new_words)
    new_words = sentence.split(" ")

    return new_words


def get_synonyms(word: str) -> List[str]:
    synonyms = set(
        [
            x.replace("_", " ").replace("-", " ")
            for synset in wordnet.synsets(word)
            for x in synset.lemma_names()
        ]
    )
    if len(synonyms) == 0:
        return [word]
    else:
        return list(synonyms)


def random_deletion(words: List[str], p: float) -> List[str]:
    """Randomly delete words from the sentence with probability p"""

    # obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

    # randomly delete words with probability p
    new_words = []
    for word in words:
        if random.uniform(0, 1) > p:
            new_words.append(word)

    # if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        return [words[random.randint(0, len(words) - 1)]]

    return new_words


def random_swap(words: List[str], n: int) -> List[str]:
    """Randomly swap two words in the sentence n times"""
    new_words = words.copy()
    for _ in range(n):
        idx1 = random.randint(0, len(new_words) - 1)
        idx2, counter = idx1, 0
        while (counter <= 3) and (idx2 == idx1):
            idx2 = random.randint(0, len(new_words) - 1)
            counter += 1
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return new_words


def random_insertion(words: List[str], n: int) -> List[str]:
    """Randomly insert n words into the sentence"""
    new_words = words.copy()
    for _ in range(n):
        synonyms, counter = [], 0
        while (counter < 10) and (len(synonyms) < 1):
            random_word = new_words[random.randint(0, len(new_words) - 1)]
            synonyms = get_synonyms(random_word)
            counter += 1
        new_words.insert(random.randint(0, len(new_words) - 1), synonyms[0])
    return new_words


def eda(
    sentence: str,
    alpha_sr: float = 0.1,
    alpha_ri: float = 0.1,
    alpha_rs: float = 0.1,
    p_rd: float = 0.1,
    num_aug: int = 9,
) -> List[str]:
    words = nltk.word_tokenize(sentence)
    detokenizer = TreebankWordDetokenizer()
    num_words = len(words)
    augmented_sentences, num_new_per_technique = [], int(num_aug / 4) + 1

    # sr
    if alpha_sr > 0:
        n_sr = max(1, int(alpha_sr * num_words))
        for _ in range(num_new_per_technique):
            a_words = synonym_replacement(words, n_sr)
            augmented_sentences.append(detokenizer.detokenize(a_words))

    # ri
    if alpha_ri > 0:
        n_ri = max(1, int(alpha_ri * num_words))
        for _ in range(num_new_per_technique):
            a_words = random_insertion(words, n_ri)
            augmented_sentences.append(detokenizer.detokenize(a_words))

    # rs
    if alpha_rs > 0:
        n_rs = max(1, int(alpha_rs * num_words))
        for _ in range(num_new_per_technique):
            a_words = random_swap(words, n_rs)
            augmented_sentences.append(detokenizer.detokenize(a_words))

    # rd
    if p_rd > 0:
        for _ in range(num_new_per_technique):
            a_words = random_deletion(words, p_rd)
            augmented_sentences.append(detokenizer.detokenize(a_words))

    augmented_sentences = list(filter(lambda x: len(x) > 0, augmented_sentences))
    shuffle(augmented_sentences)

    # trim so that we have the desired number of augmented sentences
    if len(augmented_sentences) > num_aug:
        augmented_sentences = random.sample(augmented_sentences, k=num_aug)

    return augmented_sentences
