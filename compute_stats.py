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
print(pairs)
