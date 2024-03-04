import numpy as np
import sys
import argparse
import json

def reader(options):
    data = []
    with open(options['pred_file_path'], 'r') as f:
        for line in f:
            data.append(line.strip('\n'))

    return data

def process(data, options):
    data_ = []
    for sample in data:
        if options['dataset'] == 'formal2informal':
            data_.append({'text':sample, 'formality': "informal"})
        elif options['dataset'] == 'informal2formal':
            data_.append({'text':sample, 'formality': "formal"})
        elif options['dataset'] == 'formality_transfer':
                data_.append({'text':sample, 'formality': "formal"})
    save_path = options['pred_file_path'].replace('.txt', '.json')
    with open(save_path, 'w') as f:
        for sample in data_:
            json.dump(sample, f)
            f.write('\n')

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-file-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    options = vars(parser.parse_args(args=argv))
    print(options)

    data = reader(options)
    process(data, options)
	




if __name__ == "__main__":
    main(sys.argv[1:])
