from evaluate import load
import numpy as np
import sys
import argparse

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-file-path", type=str, required=True)
    parser.add_argument("--ref-file-path", type=str, required=True)
    options = vars(parser.parse_args(args=argv))
    print(options)

    bert_score = bert_score_cal(options['ref_file_path'], options['gen_file_path'])
    print('Bert score:', np.mean(bert_score["precision"]))


def bert_score_cal(ref_path, gen_path):
    refs = []
    gens = []
    with open(ref_path, "r") as f:
        for line in f:
            refs.append(line.strip("\n"))
    with open(gen_path, "r") as f:
        for line in f:
            gens.append(line.strip("\n"))
    bertscore = load("bertscore")
    score = bertscore.compute(predictions=gens, references=refs, lang="en")
    return score
    


if __name__ == "__main__":
    main(sys.argv[1:])