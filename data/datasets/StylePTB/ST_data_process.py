import os
import json
import random

train_tense_adjadv_removal_path= "Compositional_Datasets/Tense_ADJADV_Removal/train.tsv"
test_tense_adjadv_removal_path= "Compositional_Datasets/Tense_ADJADV_Removal/test.tsv"

train_tense_pp_frontback_path = "Compositional_Datasets/Tense_PP_Front_Back/train.tsv"
test_tense_pp_frontback_path = "Compositional_Datasets/Tense_PP_Front_Back/test.tsv"

train_tense_pp_removal_path = "Compositional_Datasets/Tense_PP_Removal/train.tsv"
test_tense_pp_removal_path = "Compositional_Datasets/Tense_PP_Removal/test.tsv"

train_tense_voice_path = "Compositional_Datasets/Tense_Voice/train.tsv"
test_tense_voice_path = "Compositional_Datasets/Tense_Voice/test.tsv"

def tense_adjadv_reader(path, save_path):
    '''
    Tense: 
    0: uncerten, means that dont change the tense, but need to determine the tense of source sentence
    1: to future, means that the target sentence is in future tense
    2: to past, means that the target sentence is in past tense
    3: to present, means that the target sentence is in present tense

    ADJADV removal:
    0: dont remove the adjs and advs
    1: remove
    '''
    source_data = []
    target_data = []

    with open(path, 'r') as f:
        source_sents = {}
        for line in f:
            line = line.strip("\n")
            line = line.split("\t")
            
            #target samples
            target_style  = [int(line[0].split(" ")[:2][0])-1, int(line[0].split(" ")[:2][1])]
            if target_style == [-1, 0]:
                continue
            # target_data.append({'sentence': line[1], 'style_label': target_style})

            source_sent = " ".join(line[0].split(" ")[2:])
            if source_sent not in source_sents:
                source_sents[source_sent] = {"tense": set([target_style[0]]), "remove": set([target_style[1]]), "target":[{'sentence': line[1].lower(), 'style_label': target_style}]}
            else:
                source_sents[source_sent]["tense"].add(target_style[0])
                source_sents[source_sent]["remove"].add(target_style[1])
                source_sents[source_sent]["target"].append({'sentence': line[1].lower(), 'style_label': target_style})
            
    for sent in source_sents.keys():
        '''
        special cases:
        0 1
        3 *
        '''
        # source removal style
        if 1 in source_sents[sent]["remove"]:
            source_remove = 0
        else:
            source_remove = 1
        # source tense style
        tense_set = set([0, 1, 2])
        if len(list(tense_set-source_sents[sent]["tense"])) != 0:
            source_tense = random.sample(list(tense_set-source_sents[sent]["tense"]), 1)[0]
            source_data.append({'sentence': sent.lower(), 'style_label': [source_tense, source_remove]})
            for sample in source_sents[sent]["target"]:
                if source_remove == 1:
                    sample["style_label"][1] = 1
                if sample["style_label"][0] == -1:
                    sample["style_label"][0] = source_tense
                target_data.append(sample)
        else:
            source_tense = 2
            source_data.append({'sentence': sent.lower(), 'style_label': [source_tense, source_remove]})
            for sample in source_sents[sent]["target"]:
                if sample["style_label"][0] == source_tense:
                    continue
                if source_remove == 1:
                    sample["style_label"][1] = 1
                if sample["style_label"][0] == -1:
                    sample["style_label"][0] = source_tense
                target_data.append(sample)
                    

        

    data = source_data + target_data

    print("The number of samples: {}".format(len(data)))
    with open(save_path, 'w') as f:
        for sample in data:
            json.dump(sample, f)
            f.write('\n')

def tense_pp_reader(path, save_path):
    '''
    Tense: 
    0: uncerten, means that don't change the tense, but need to determine the tense of source sentence
    1: to future, means that the target sentence is in future tense
    2: to past, means that the target sentence is in past tense
    3: to present, means that the target sentence is in present tense

    PP Front Back:
    0: uncerten, means that don't change the PP position, but need to determine the position of the source sentence
    1: to PP back, the PP position of the target sentence is back
    2: to PP front, the PP position of the target sentence is front
    '''
    source_data = []
    target_data = []

    with open(path, 'r') as f:
        source_sents = {}
        for line in f:
            line = line.strip("\n")
            line = line.split("\t")
            
            #target samples
            target_style  = [int(line[0].split(" ")[:2][0])-1, int(line[0].split(" ")[:2][1])-1]
            if target_style == [-1, -1]:
                continue
            # target_data.append({'sentence': line[1], 'style_label': target_style})

            source_sent = " ".join(line[0].split(" ")[2:])
            if source_sent not in source_sents:
                source_sents[source_sent] = {"tense": set([target_style[0]]), "PP": set([target_style[1]]), "target":[{'sentence': line[1].lower(), 'style_label': target_style}]}
            else:
                source_sents[source_sent]["tense"].add(target_style[0])
                source_sents[source_sent]["PP"].add(target_style[1])
                source_sents[source_sent]["target"].append({'sentence': line[1].lower(), 'style_label': target_style})
            
    for sent in source_sents.keys():
        '''
        special cases:
        0 1
        3 *
        '''
        # source removal style
        pp_set = set([0,1])
        if len(list(pp_set-source_sents[sent]["PP"])) != 0:
            source_pp = random.sample(list(pp_set-source_sents[sent]["PP"]), 1)[0]
        
        # source tense style
        tense_set = set([0, 1, 2])
        if len(list(tense_set-source_sents[sent]["tense"])) != 0:
            source_tense = random.sample(list(tense_set-source_sents[sent]["tense"]), 1)[0]
            source_data.append({'sentence': sent.lower(), 'style_label': [source_tense, source_pp]})
            for sample in source_sents[sent]["target"]:
                if sample["style_label"][1] == -1:
                    sample["style_label"][1] = source_pp
                if sample["style_label"][0] == -1:
                    sample["style_label"][0] = source_tense
                target_data.append(sample)
        else:
            source_tense = 2
            source_data.append({'sentence': sent.lower(), 'style_label': [source_tense, source_pp]})
            for sample in source_sents[sent]["target"]:
                if sample["style_label"][0] == source_tense:
                    continue
                if sample["style_label"][1] == -1:
                    sample["style_label"][1] = source_pp
                if sample["style_label"][0] == -1:
                    sample["style_label"][0] = source_tense
                target_data.append(sample)
                    

        

    data = source_data + target_data
    
    if not os.path.exists("/".join(save_path.split("/")[:2])):
        os.mkdir("/".join(save_path.split("/")[:2]))
    if not os.path.exists("/".join(save_path.split("/")[:3])):
        os.mkdir("/".join(save_path.split("/")[:3]))
    print("The number of samples: {}".format(len(data)))
    with open(save_path, 'w') as f:
        for sample in data:
            json.dump(sample, f)
            f.write('\n')

def tense_pp_removal_reader(path, save_path):
    '''
    Tense: 
    0: uncerten, means that dont change the tense, but need to determine the tense of source sentence
    1: to future, means that the target sentence is in future tense
    2: to past, means that the target sentence is in past tense
    3: to present, means that the target sentence is in present tense

    ADJADV removal:
    0: dont remove the adjs and advs
    1: remove
    '''
    source_data = []
    target_data = []

    with open(path, 'r') as f:
        source_sents = {}
        for line in f:
            line = line.strip("\n")
            line = line.split("\t")
            
            #target samples
            target_style  = [int(line[0].split(" ")[:2][0])-1, int(line[0].split(" ")[:2][1])-4]
            if target_style == [-1, 0]:
                continue
            # target_data.append({'sentence': line[1], 'style_label': target_style})

            source_sent = " ".join(line[0].split(" ")[2:])
            if source_sent not in source_sents:
                source_sents[source_sent] = {"tense": set([target_style[0]]), "remove": set([target_style[1]]), "target":[{'sentence': line[1].lower(), 'style_label': target_style}]}
            else:
                source_sents[source_sent]["tense"].add(target_style[0])
                source_sents[source_sent]["remove"].add(target_style[1])
                source_sents[source_sent]["target"].append({'sentence': line[1].lower(), 'style_label': target_style})
            
    for sent in source_sents.keys():
        '''
        special cases:
        0 1
        3 *
        '''
        # source removal style
        if 1 in source_sents[sent]["remove"]:
            source_remove = 0
        else:
            source_remove = 1
        # source tense style
        tense_set = set([0, 1, 2])
        if len(list(tense_set-source_sents[sent]["tense"])) != 0:
            source_tense = random.sample(list(tense_set-source_sents[sent]["tense"]), 1)[0]
            source_data.append({'sentence': sent.lower(), 'style_label': [source_tense, source_remove]})
            for sample in source_sents[sent]["target"]:
                if source_remove == 1:
                    sample["style_label"][1] = 1
                if sample["style_label"][0] == -1:
                    sample["style_label"][0] = source_tense
                target_data.append(sample)
        else:
            source_tense = 2
            source_data.append({'sentence': sent.lower(), 'style_label': [source_tense, source_remove]})
            for sample in source_sents[sent]["target"]:
                if sample["style_label"][0] == source_tense:
                    continue
                if source_remove == 1:
                    sample["style_label"][1] = 1
                if sample["style_label"][0] == -1:
                    sample["style_label"][0] = source_tense
                target_data.append(sample)
                      

    data = source_data + target_data

    if not os.path.exists("/".join(save_path.split("/")[:2])):
        os.mkdir("/".join(save_path.split("/")[:2]))
    if not os.path.exists("/".join(save_path.split("/")[:3])):
        os.mkdir("/".join(save_path.split("/")[:3]))

    print("The number of samples: {}".format(len(data)))
    with open(save_path, 'w') as f:
        for sample in data:
            json.dump(sample, f)
            f.write('\n')

def tense_voice_reader(path, save_path):
    '''
    Tense: 
    0: uncerten, means that don't change the tense, but need to determine the tense of source sentence
    1: to future, means that the target sentence is in future tense
    2: to past, means that the target sentence is in past tense
    3: to present, means that the target sentence is in present tense

    Voice:
    0: uncerten, means that don't change the voice, but need to determine the voice of the source sentence
    1: to Passive, the voice of the target sentence is passive
    2: to Active, the Voice of the target sentence is active
    '''
    source_data = []
    target_data = []

    with open(path, 'r') as f:
        source_sents = {}
        for line in f:
            line = line.strip("\n")
            line = line.split("\t")
            
            #target samples
            target_style  = [int(line[0].split(" ")[:2][0])-1, int(line[0].split(" ")[:2][1])-1]
            if target_style == [-1, -1]:
                continue
            # target_data.append({'sentence': line[1], 'style_label': target_style})

            source_sent = " ".join(line[0].split(" ")[2:])
            if source_sent not in source_sents:
                source_sents[source_sent] = {"tense": set([target_style[0]]), "voice": set([target_style[1]]), "target":[{'sentence': line[1].lower(), 'style_label': target_style}]}
            else:
                source_sents[source_sent]["tense"].add(target_style[0])
                source_sents[source_sent]["voice"].add(target_style[1])
                source_sents[source_sent]["target"].append({'sentence': line[1].lower(), 'style_label': target_style})
            
    for sent in source_sents.keys():
        '''
        special cases:
        0 1
        3 *
        '''
        # source removal style
        pp_set = set([0,1])
        if len(list(pp_set-source_sents[sent]["voice"])) != 0:
            source_voice = random.sample(list(pp_set-source_sents[sent]["voice"]), 1)[0]
        
        # source tense style
        tense_set = set([0, 1, 2])
        if len(list(tense_set-source_sents[sent]["tense"])) != 0:
            source_tense = random.sample(list(tense_set-source_sents[sent]["tense"]), 1)[0]
            source_data.append({'sentence': sent.lower(), 'style_label': [source_tense, source_voice]})
            for sample in source_sents[sent]["target"]:
                if sample["style_label"][1] == -1:
                    sample["style_label"][1] = source_voice
                if sample["style_label"][0] == -1:
                    sample["style_label"][0] = source_tense
                target_data.append(sample)
        else:
            source_tense = 2
            source_data.append({'sentence': sent.lower(), 'style_label': [source_tense, source_voice]})
            for sample in source_sents[sent]["target"]:
                if sample["style_label"][0] == source_tense:
                    continue
                if sample["style_label"][1] == -1:
                    sample["style_label"][1] = source_voice
                if sample["style_label"][0] == -1:
                    sample["style_label"][0] = source_tense
                target_data.append(sample)
                    

        

    data = source_data + target_data
    
    if not os.path.exists("/".join(save_path.split("/")[:2])):
        os.mkdir("/".join(save_path.split("/")[:2]))
    if not os.path.exists("/".join(save_path.split("/")[:3])):
        os.mkdir("/".join(save_path.split("/")[:3]))
    print("The number of samples: {}".format(len(data)))
    with open(save_path, 'w') as f:
        for sample in data:
            json.dump(sample, f)
            f.write('\n')

# tense_adjadv_reader(train_tense_adjadv_removal_path, "adapterTST/tense_adjadv_removal/train/style_transfer_unsup.json")
# tense_adjadv_reader(test_tense_adjadv_removal_path, "adapterTST/tense_adjadv_removal/test/style_transfer_unsup.json")

# tense_pp_reader(train_tense_pp_frontback_path, "adapterTST/tense_pp_front_back/train/style_transfer_unsup.json")
# tense_pp_reader(test_tense_pp_frontback_path, "adapterTST/tense_pp_front_back/test/style_transfer_unsup.json")

# tense_pp_removal_reader(train_tense_pp_removal_path, "adapterTST/tense_pp_removal/train/style_transfer_unsup.json")
# tense_pp_removal_reader(test_tense_pp_removal_path, "adapterTST/tense_pp_removal/test/style_transfer_unsup.json")

tense_voice_reader(train_tense_voice_path, "adapterTST/tense_voice/train/style_transfer_unsup.json")
tense_voice_reader(test_tense_voice_path, "adapterTST/tense_voice/test/style_transfer_unsup.json")