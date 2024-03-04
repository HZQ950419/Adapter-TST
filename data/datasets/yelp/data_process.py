import json

train_path = 'train/train.txt'
test_path = 'test/test.txt'

def reader(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def process(data, save_path):
    data_ = []
    for sample in data:
        if sample['score'] == 0:
            data_.append({'text': sample['review'], 'sentiment': "negative"})
        if sample['score'] == 1:
            data_.append({'text': sample['review'], 'sentiment': "positive"})
    print("The number of samples: {}".format(len(data_)))
    with open(save_path, 'w') as f:
        for sample in data_:
            json.dump(sample, f)
            f.write('\n')


train = reader(train_path)
test = reader(test_path)
process(train, 'train/train_cls.json')
process(test, 'test/test_cls.json')



