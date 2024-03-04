import numpy as np
import sys
import argparse

def reader(path):
    data = []
    preds = []
    with open(path, 'r') as f:
        for line in f:
            data.append(line.strip('\n'))
    for sample in data:
        if sample == 'informal' or sample == 'negative':
            preds.append(0)
        elif sample == 'formal' or sample == 'positive':
            preds.append(1)
        else:
            preds.append(-1)
    return preds

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-file-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    options = vars(parser.parse_args(args=argv))
    print(options)

    preds = reader(options['pred_file_path'])
    if options['dataset'] == 'sentiment':
        labels = [1 for i in range(76392)] + [0 for i in range(50278)]
    elif options['dataset'] == 'formality':
        labels = [0 for i in range(1332)] + [1 for i in range(1019)]
    elif options['dataset'] == 'formal2informal':
        labels = [0 for i in range(1019)]
    elif options['dataset'] == 'informal2formal':
        labels = [1 for i in range(1332)]
    elif options['dataset'] == 'formality_transfer':
        labels = [1 for i in range(1332)] + [0 for i in range(1019)]
    
    assert len(preds) == len(labels), 'The length of prediction doesn\'t equal to labels!'
    corrects = np.equal(preds, labels) 
    acc = np.sum(corrects)/len(labels)
    print('Correct predictions: {}\nAccuracy: {}'.format(np.sum(corrects), acc))
        


if __name__ == "__main__":
    main(sys.argv[1:])
