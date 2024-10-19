
import os
import torch
import json
import re
import numpy as np
import gc
gc.collect()

current_directory = os.getcwd()
train_JSON_path = current_directory+'/MLDS_hw2_1_data/training_label.json'
train_feature_path = current_directory+'/MLDS_hw2_1_data/training_data/feat'
test_JSON_path = current_directory+'/MLDS_hw2_1_data/testing_label.json'
test_feature_path = current_directory+'/MLDS_hw2_1_data/testing_data/feat'

class data_pro_train():
    def __init__(self,json_path,feature_path,min_word_count=3):
        self.json_path = json_path
        self.feature_path = feature_path
        self.min_word_count = min_word_count
        self.json_file = None
        self.word_count = {}
        self.good_words = []
        self.translations = {}
        self.words_to_index = {}
        self.annoted_captions = []
        self.feature_dictionary = {}
        
        self.getJSONFile()
        self.getWordCountDic()
        self.getWordMap()
        self.wordAnnotation()
        self.featureDic()
        
    def getJSONFile(self):
        with open(self.json_path, 'r') as f:
            file = json.load(f)
        self.json_file = file
        
    def getWordCountDic(self):
        count={}
        for line in self.json_file:
            for jsentence in line['caption']:
                sentence = re.sub('[.!,;?-]]', ' ', jsentence).split()
                for word in sentence:
                    word = word.replace('.', '') if '.' in word else word
                    word = word.lower()
                    word = re.sub(r"[^a-zA-Z0-9]+", '', word)
                    if word in count:
                        count[word] += 1
                    else:
                        count[word] = 1
                    
        self.word_count = count
        
    def getWordMap(self):
        goodWords = [k for k, v in self.word_count.items() if v > self.min_word_count]
        tokens = [('<PAD>', 0), ('<SOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
        indexWord = {i + len(tokens): w for i, w in enumerate(goodWords)}
        wordIndex = {w: i + len(tokens) for i, w in enumerate(goodWords)}
        for token, index in tokens:
            indexWord[index] = token
            wordIndex[token] = index
    
        self.good_words = goodWords
        self.translations = indexWord
        self.words_to_index = wordIndex
        
    def replaceWords(self,sentence):
        sentence = re.sub(r'[.!,;?]', ' ', sentence).split()
        sentence = ['<SOS>'] + [w if (self.word_count.get(w, 0) > self.min_word_count) \
                                    else '<UNK>' for w in sentence] + ['<EOS>']
        sentence = [self.words_to_index[w] for w in sentence]
        return sentence
    
    def wordAnnotation(self):
        dataCombined = []
        for directory in self.json_file:
            for sente in directory['caption']:
                sentence = self.replaceWords(sente.lower())
                dataCombined.append((directory['id'], sentence))
        self.annoted_captions = dataCombined
        
    def featureDic(self):
        featureDictonary={}
        files = os.listdir(self.feature_path)
        for file in files:
                key = file.split('.npy')[0]
                value = np.load(os.path.join(self.feature_path, file),allow_pickle=True)
                featureDictonary[key] = value
        self.feature_dictionary = featureDictonary
        
        
    def __len__(self):
        return len(self.annoted_captions)
    
    def __getitem__(self, idx):
        assert (idx < self.__len__())
        avi_file_name, sentence = self.annoted_captions[idx]
        data = torch.Tensor(self.feature_dictionary[avi_file_name])
        data += torch.Tensor(data.size()).random_(0, 2000)/10000.
        return torch.Tensor(data), torch.Tensor(sentence)
    


class data_pro_test():
    def __init__(self,feature_path):
        self.feature_path = feature_path
        self.test_feature_dic = []
        
        self.testFeatureDic()
        
    def testFeatureDic(self):
        files = os.listdir(self.feature_path)
        for file in files:
            key = file.split('.npy')[0]
            value = np.load(os.path.join(self.feature_path, file),allow_pickle=True)
            self.test_feature_dic.append([key, value])
            
    def __len__(self):
        return len(self.test_feature_dic)
    def __getitem__(self, idx):
        return self.test_feature_dic[idx]
               
def createMiniBatch(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    avi_data, captions = zip(*data) 
    avi_data = torch.stack(avi_data, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return avi_data, targets, lengths
