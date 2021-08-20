import random


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


