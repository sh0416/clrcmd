import random
from functools import partial


def swap_word_middle(w: str, seed: int) -> str:
    """첫 문자와 마지막 문자를 제외한 문자들 중 인접한 두 문자를 바꾼다
    
    :param w: 바꿀 단어
    :type w: str
    :param seed: 랜덤 함수에 쓸 상태
    :type seed: int
    :return: 바뀐 단어
    :rtype: str
    """
    if len(w) > 3:
        w = list(w)
        random.seed(seed)
        i = random.randint(1, len(w) - 3)
        w[i], w[i + 1] = w[i + 1], w[i]
        w = "".join(w)
    return w


def permute_word(w: str, seed: int) -> str:
    """단어에 있는 문자 모두 임의로 섞는다

    :param w: 바꿀 단어
    :type w: str
    :param seed: 랜덤 함수에 쓸 시드
    :type seed: int
    :return: 바뀐 단어
    :rtype: str
    """
    w = list(w)
    random.seed(seed)
    random.shuffle(w)
    w = "".join(w)
    return w


def permute_word_middle(w: str, seed: int) -> str:
    """첫 문자와 마지막 문자를 제외한 문자들을 임의로 섞는다

    :param w: 바꿀 단어
    :type w: str
    :param seed: 랜덤 함수에 쓸 상태
    :type seed: int
    :return: 바뀐 단어
    :rtype: str
    """    
    if len(w) > 3:
        middle = list(w[1:-1])
        random.seed(seed)
        random.shuffle(middle)
        middle= "".join(middle)
        w = w[0] + middle + w[-1]
    return w 

    
qwerty_ch2ch = {}
with open("res/en.key") as f:
    for line in f:
        src, tgt = line.split()
        qwerty_ch2ch[src] = tgt


def retrieve_qwerty_typo(w: str, seed: int) -> str:
    """

    :param w: 바꿀 단어
    :type w: str
    :param seed: 랜덤 함수에 쓸 상태
    :type seed: int
    :return: 바뀐 단어
    :rtype: str
    """    
    w = list(w)
    random.seed(seed)
    i = random.randint(0, len(w) - 1)
    candidate = qwerty_ch2ch.get(w[i].lower(), [w[i]])
    if w[i].isupper():
        w[i] = random.choice(candidate).upper()
    else:
        w[i] = random.choice(candidate)
    return "".join(w)


natural_word2word = {}
with open("res/en.natural") as f:
    for line in f:
        src, tgt = line.split()
        natural_word2word[src] = tgt

def retrieve_natural_typo(w: str, seed: int) -> str:
    """

    :param w: 바꿀 단어
    :type w: str
    :param seed: 랜덤 함수에 쓸 상태
    :type seed: int
    :return: 바뀐 단어
    :rtype: str
    """    
    random.seed(seed)
    candidate = natural_word2word.get(w, [w]) 
    return random.choice(candidate)


def apply_sequence(wlist: List[str], strategy: str, seed: int) -> List[str]: 
    if strategy == "swap":
        strategy = swap_word
    elif strategy == "permute":
        strategy = permute_word
    elif strategy == "permute_middle":
        strategy = permute_word_middle
    elif strategy == "qwerty":
        strategy = retrieve_qwerty_typo
    elif strategy == "natural":
        strategy = retrieve_natural_typo
    random.seed(seed)
    seed_list = random.choices(range(10000), k=len(wlist))
    return list(map(lambda x: strategy(x[0], x[1]), zip(wlist, seed_list)))

