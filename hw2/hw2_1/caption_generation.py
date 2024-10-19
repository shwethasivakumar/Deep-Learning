
import sys
import os
import torch
import json
from torch.utils.data import DataLoader
from torch.autograd import Variable
from train import MODELS, encoderRNN, decoderRNN, attention
from pretrain import data_pro_test
from bleu_eval import BLEU


def captionGeneration(test_loader):
    model = torch.load('SavedModel/model11.h5', map_location=lambda storage, loc: storage)
    model = model.cuda()
    model.eval()
    final_out = []
    with open('translations.json', 'r') as fp:
        translations = json.load(fp)
        
    translations = {int(k):v for k,v in translations.items()}
    for batch_idx, batch in enumerate(test_loader):
        id, test_features = batch
        test_features = test_features.cuda()
        id, test_features = id, Variable(test_features).float()

        seq_logProb, caption_predictions = model(test_features, mode='inference')
        result = [[translations[x.item()] if translations[x.item()] != '<UNK>' else 'something' for x in s] for s in caption_predictions]
        result = [' '.join(s).split('<EOS>')[0] for s in result]
        video_caption = zip(id, result)
        for vc in video_caption:
            final_out.append(vc)
    return final_out


if __name__ == "__main__":
    filepath = os.getcwd()+sys.argv[1]
    if not os.path.exists(filepath):
        filepath = sys.argv[1]
        
    print('Test Feature path - ',filepath)
    dataset = data_pro_test(filepath)
    testing_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)   
    predicted_cap = captionGeneration(testing_loader)
    caption_result = {}
    with open(sys.argv[2], 'w') as f:
        for id, caption in predicted_cap:
            caption_result[id]=caption
            f.write('{},{}\n'.format(id, caption))
    test_JSON_path = ''.join(filepath.rpartition('/')[:1])
    test_JSON_path = ''.join(test_JSON_path.rpartition('/')[:1])
    
    if not os.path.exists(test_JSON_path+'/testing_label.json'):
        test_JSON_path = '/testing_label.json'
        if not os.path.exists(test_JSON_path):
            print('Cannot find JSON label data for BLEU evaluation')
    print('Test JSON path - ',test_JSON_path+'/testing_label.json')
    test_json = json.load(open(test_JSON_path+'/testing_label.json'))
     
    bleu=[]
    for item in test_json:
        score_per_video = []
        captions = [x.rstrip('.') for x in item['caption']]
        score_per_video.append(BLEU(caption_result[item['id']],captions,True))
        bleu.append(score_per_video[0])
    average = sum(bleu) / len(bleu)
    print("Average bleu score is " + str(average))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    