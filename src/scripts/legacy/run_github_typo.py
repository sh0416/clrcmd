import csv
import json
import re

from bs4 import BeautifulSoup

_RE_COMBINE_WHITESPACE = re.compile(r"\s+")


def extract_text(s: str) -> str:
    soup = BeautifulSoup(s)
    return _RE_COMBINE_WHITESPACE.sub(" ", soup.get_text(separator=" ")).strip()


data = []
with open("/nas/home/sh0416/data/github-typo-corpus.v1.0.0.jsonl") as f:
    for line in f:
        row = json.loads(line)
        for edit in row["edits"]:
            x = extract_text(edit["tgt"]["text"])
            x_pos = extract_text(edit["src"]["text"])
            if x == x_pos or len(x) == 0 or len(x_pos) == 0:
                continue
            data.append((x, x_pos))
data = list(set(data))

with open("/nas/home/sh0416/data/github-typo-corpus-contrastive.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(data)
