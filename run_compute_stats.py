import itertools
import statistics

from evaluate import load_sts12, load_sts13, load_sts14, load_sts15, load_sts16

sts = [
    load_sts12("data/STS/STS12-en-test"),
    load_sts13("data/STS/STS13-en-test"),
    load_sts14("data/STS/STS14-en-test"),
    load_sts15("data/STS/STS15-en-test"),
    load_sts16("data/STS/STS16-en-test"),
]


# 쌍 개수 구하기
pairs = [row for dataset in sts for source in dataset.values() for row in source]
print(f"The number of pairs {len(pairs)}")

stream1 = map(lambda x: x.input.text1, pairs)
stream2 = map(lambda x: x.input.text2, pairs)
sentences = list(itertools.chain(stream1, stream2))
assert all(map(lambda x: len(x) > 0, sentences))
print(f"The number of sentences: {len(sentences)}")

lengths_token = list(map(len, sentences))
print(f"Minimum length: {min(lengths_token)}")
print(f"Maximum length: {max(lengths_token)}")
print(f"Average length: {statistics.mean(lengths_token)}")

lengths_char = list(map(lambda x: len(" ".join(x)), sentences))
print(f"Minimum length: {min(lengths_char)}")
print(f"Maximum length: {max(lengths_char)}")
print(f"Average length: {statistics.mean(lengths_char)}")