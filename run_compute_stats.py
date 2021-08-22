import itertools
import statistics

from dataset import load_sts12, load_sts13, load_sts14, load_sts15, load_sts16

sts = {
    "sts12": load_sts12("data/STS/STS12-en-test"),
    "sts13": load_sts13("data/STS/STS13-en-test"),
    "sts14": load_sts14("data/STS/STS14-en-test"),
    "sts15": load_sts15("data/STS/STS15-en-test"),
    "sts16": load_sts16("data/STS/STS16-en-test"),
}


# 쌍 개수 구하기
pairs = [
    row
    for dataset in sts.values()
    for source in dataset.values()
    for row in source
]
print(f"The number of pairs {len(pairs)}")

stream1 = map(lambda x: x.input.text1, pairs)
stream2 = map(lambda x: x.input.text2, pairs)
sentences = list(itertools.chain(stream1, stream2))
assert all(map(lambda x: len(x) > 0, sentences))
print(f"The number of sentences: {len(sentences)}")

lengths_token = list(map(len, sentences))
print(f"Minimum length (level-word): {min(lengths_token)}")
print(f"Maximum length (level-word): {max(lengths_token)}")
print(f"Average length (level-word): {statistics.mean(lengths_token)}")

lengths_char = list(map(lambda x: len(" ".join(x)), sentences))
print(f"Minimum length (level-char): {min(lengths_char)}")
print(f"Maximum length (level-char): {max(lengths_char)}")
print(f"Average length (level-char): {statistics.mean(lengths_char)}")

# 각 데이터셋 별 문자 개수
for k, v in sts.items():
    dataset_chars = 0
    for k2, v2 in v.items():
        total_chars = sum(map(lambda x: len(" ".join(x.input.text1)), v2))
        total_chars += sum(map(lambda x: len(" ".join(x.input.text2)), v2))
        dataset_chars += total_chars
        print(k, k2, total_chars)
    print(dataset_chars)
