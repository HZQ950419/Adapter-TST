import json
from nltk.tokenize.treebank import TreebankWordDetokenizer

train_path = ["train/train.0", "train/train.1"]
test_path = ["test/test.0", "test/test.1"]


def reader(path_list, test=False):
    neg, pos = [], []
    for path in path_list:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip("\n")
                if path[-1] == "0":
                    neg.append(line)
                if path[-1] == "1":
                    pos.append(line)
    return neg, pos

def process_transfer_unsup(source, target, save_path):
    data_ = []
    #assert len(source) == len(target), 'Length Error!'
    for i in range(len(source)):
        data_.append({'sentence':TreebankWordDetokenizer().detokenize(source[i].split(" ")), 'style_label': 0})
    for i in range(len(target)):
        data_.append({'sentence':TreebankWordDetokenizer().detokenize(target[i].split(" ")), 'style_label': 1})
    print("The number of samples: {}".format(len(data_)))
    with open(save_path, 'w') as f:
        for sample in data_:
            json.dump(sample, f)
            f.write('\n')

train_neg, train_pos = reader(train_path)
test_neg, test_pos = reader(test_path)

process_transfer_unsup(train_neg, train_pos, 'train/sentiment_transfer_unsup.json')
process_transfer_unsup(test_neg, test_pos, 'test/sentiment_transfer_unsup.json')