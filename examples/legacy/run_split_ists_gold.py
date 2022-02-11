import argparse

from sentsim.eval.ists import AlignmentPair, load_alignment, save_alignment

parser = argparse.ArgumentParser()
parser.add_argument(
    "--alignment-path",
    type=str,
    default="/nas/home/sh0416/data/semeval16/task2/test_goldStandard/STSint.testinput.images.wa",
)
parser.add_argument("--remove-oppo", action="store_true")


def main():
    args = parser.parse_args()
    gold_alignments = load_alignment(args.alignment_path)
    for alignment in gold_alignments:
        words1 = alignment["sent1"].split()
        words2 = alignment["sent2"].split()
        if words1[-1] == "." and words2[-1] == ".":
            alignment["pairs"].append(
                AlignmentPair(
                    sent1_word_ids=[len(words1)],
                    sent2_word_ids=[len(words2)],
                    type="EQUI",
                    score=5.0,
                    comment=". == .",
                )
            )

    for alignment in gold_alignments:
        removed_type = ["NOALI", "OPPO"] if args.remove_oppo else ["NOALI"]
        alignment["pairs"] = [
            x for x in alignment["pairs"] if x["type"] not in removed_type and x["score"] >= 3.0
        ]

    outfile = f"{args.alignment_path}.equi"
    if args.remove_oppo:
        outfile += ".nooppo"
    save_alignment(gold_alignments, outfile)


if __name__ == "__main__":
    main()
