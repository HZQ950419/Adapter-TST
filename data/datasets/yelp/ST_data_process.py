import json
from nltk.tokenize.treebank import TreebankWordDetokenizer

train_path = "train/train.txt"
test_path = "test/test.txt"

def reader(path, test=False):
    neg, pos = [], []
    with open(path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            if sample["score"] == 0:
                neg.append(sample["review"])
            if sample["score"] == 1:
                pos.append(sample["review"])
    if test:
        return neg, pos
    else:
        return neg, pos

def process_for2inf(data_inf, data_for, save_path):
    data_ = []
    for i in range(len(data_inf)):
        data_.append({'formal': data_for[i], 'informal': data_inf[i]})
    print("The number of samples: {}".format(len(data_)))
    with open(save_path, 'w') as f:
        for sample in data_:
            json.dump(sample, f)
            f.write('\n')

def process_formality_transfer(source, target, save_path):
    data_ = []
    assert len(source) == len(target), 'Length Error!'
    for i in range(len(source)):
        data_.append({'source':source[i], 'target':target[i]})
    print("The number of samples: {}".format(len(data_)))
    with open(save_path, 'w') as f:
        for sample in data_:
            json.dump(sample, f)
            f.write('\n')

def process_formality_transfer_unsup(source, target, save_path):
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

def process_style_classification(source, target, save_path):
    data_ = []
    #assert len(source) == len(target), 'Length Error!'
    for i in range(len(source)):
        data_.append({'sentence1':TreebankWordDetokenizer().detokenize(source[i].split(" ")), 'label': 0})
    for i in range(len(target)):
        data_.append({'sentence1':TreebankWordDetokenizer().detokenize(target[i].split(" ")), 'label': 1})
    print("The number of samples: {}".format(len(data_)))
    with open(save_path, 'w') as f:
        for sample in data_:
            json.dump(sample, f)
            f.write('\n')


def process_test_refer(source, target, save_path):
    data_ = []
    #assert len(source) == len(target), 'Length Error!'
    for i in range(len(source)):
        data_.append(TreebankWordDetokenizer().detokenize(source[i].split(" ")))
    for i in range(len(target)):
        data_.append(TreebankWordDetokenizer().detokenize(target[i].split(" ")))
    print("The number of samples: {}".format(len(data_)))
    with open(save_path, 'w') as f:
        for sample in data_:
            f.write(sample + '\n')


def process_inf2for(data_inf, data_for, save_path):
    data_ = []
    for i in range(len(data_inf)):
        data_.append({'informal': data_inf[i], 'formal': data_for[i]})
    print("The number of samples: {}".format(len(data_)))
    with open(save_path, 'w') as f:
        for sample in data_:
            json.dump(sample, f)
            f.write('\n') 

train_neg, train_pos = reader(train_path)
test_neg, test_pos = reader(test_path, test=True)


#process_for2inf(train_inf, train_for, 'train/formal2informal.json')
#process_for2inf(test_for2inf, test_for, 'test/formal2informal.json')
#process_inf2for(train_inf, train_for, 'train/informal2formal.json')
#process_inf2for(test_inf, test_inf2for, 'test/informal2formal.json')
#process_formality_transfer(train_inf+train_for, train_for+train_inf, 'train/formality_transfer.json')
#process_formality_transfer(test_inf+test_for, test_inf2for+test_for2inf, 'test/formality_transfer.json')
process_formality_transfer_unsup(train_neg, train_pos, 'train/sentiment_transfer_unsup.json')
process_formality_transfer_unsup(test_neg, test_pos, 'test/sentiment_transfer_unsup.json')
# process_style_classification(train_neg, train_pos, 'train/sentiment_cls.json')
# process_style_classification(test_neg, test_pos, 'test/sentiment_cls.json')
# process_test_refer(test_neg, test_pos, 'test/test_500.txt')
