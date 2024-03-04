from utils.evaluator import BLEUEvaluator
import sys
import argparse

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-file-path", type=str, required=True)
    parser.add_argument("--ref-file-path", type=str, required=True)
    options = vars(parser.parse_args(args=argv))
    print(options)

    bleu_evaluator = BLEUEvaluator()
    bleu_score = bleu_evaluator.score(options['ref_file_path'], options['gen_file_path'])
    print('BLEU score:', bleu_score)

if __name__ == "__main__":
    main(sys.argv[1:])
