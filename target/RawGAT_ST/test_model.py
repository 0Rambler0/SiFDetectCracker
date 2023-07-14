from unicodedata import name
import torch
from model import RawGAT_ST
import os 
import yaml
import sys
sys.path.append('../..')
from audio_processing import ReadAudio
from torch.utils.data import DataLoader, Dataset
import numpy as np
from data_utils import *
from torchvision import transforms
from torch import Tensor


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	

class EvalDataSet(Dataset):
    def __init__(self, path):
        self.path = path
        self.file_list = os.listdir(path)
        self.length = len(self.file_list)
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        audio, samplerate = ReadAudio(os.path.join(self.path,self.file_list[idx]))
        audio = pad(audio)
        fname = self.file_list[idx]
        
        return audio, fname, samplerate

def predict(audio_path, model_path):
    dir_yaml = os.path.join(model_path, 'model_config_RawGAT_ST.yaml')
    with open(dir_yaml, 'r') as f_yaml:
            parser1 = yaml.safe_load(f_yaml)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:%s' %device)
    # device = 'cpu'
    model = RawGAT_ST(parser1['model'], device)
    model.load_state_dict(torch.load(os.path.join(model_path,'Pre_trained_models/RawGAT_ST_mul/Best_epoch.pth'), map_location=device))
    # print(model)
    model = model.to(device=device)
    model.eval()
    eval_set = EvalDataSet(audio_path)
    eval_set_loader = DataLoader(eval_set, batch_size=1, shuffle=False)
    score_list = [None]*eval_set.length
    min_score = 0
    pass_count = 0
    threshold = -0.0048
    for audio, fname, samplerate in eval_set_loader:
        audio = audio.to(device=device, dtype=torch.float)
        output = model(audio)
        score = (output[:, 1]
                       ).data.cpu().numpy().ravel().tolist()
        _, predict = output.max(dim=1)
        score = score[0]
        labels = ['fake', 'real']
        label = predict.item()
        if label == 1:
            pass_count += 1
        if score < min_score and label == 1:
            min_score = score
        score_list[int(fname[0].split('.')[0])-1] = score
    pass_rate = pass_count/len(eval_set_loader)

    return score_list, min_score, pass_rate


if __name__ == '__main__':
    path = '/home/dslab/hx/Vattack/'

    test_set = sys.argv[1]
    silence_factor = sys.argv[2]
    dir_yaml = path + 'test_model/RawGAT_ST_antispoofing/model_config_RawGAT_ST.yaml'
    with open(dir_yaml, 'r') as f_yaml:
            parser1 = yaml.safe_load(f_yaml)

    model_path = path + 'test_model/RawGAT_ST_antispoofing/Pre_trained_models/RawGAT_ST_mul/Best_epoch.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RawGAT_ST(parser1['model'], device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    # print(model)
    model.cuda()
    model.eval()
    # eval_set = EvalDataSet(path+'Nattack/audio/')
    audio_name_list = os.listdir('/home/dslab/hx/vfuzz/media/exp_data/Vattack_exp_2023_4_4/exp_long_audio/data/{}'.format(test_set))
    is_logical = ('DF' == 'logical')
    total_pass_rate = 0
    total_count = len(audio_name_list)
    pass_audio_list = []
    for audio_name in audio_name_list:
        print('audio name:%s' %audio_name)
        eval_set = EvalDataSet('/home/dslab/hx/vfuzz/media/exp_data/Vattack_exp_2023_4_4/exp_long_audio/data/{}/{}/{}/'.format(test_set, audio_name, silence_factor))
        eval_set_loader = DataLoader(eval_set, batch_size=1, shuffle=False)
        pass_count = 0
        for audio, fname, samplerate in eval_set_loader:
            audio = audio.to(device=device, dtype=torch.float)
            output = model(audio)
            score = (output[:, 1]  
                        ).data.cpu().numpy().ravel()[0]  
            value, predict = output.max(dim=1)
            if predict == 1:
                pass_count += 1
            # print('fname:%s score:%f  label:%s' %(fname[0], score, predict.item()))
        pass_rate = pass_count/len(eval_set_loader)
        total_pass_rate += pass_rate
        print('pass rate:%f' %pass_rate)
        if pass_rate > 0.8:
            pass_audio_list.append(audio_name)
    print('total pass rate:%f average_pass_rate:%f' %(total_pass_rate, total_pass_rate/total_count))
    # print('total pass rate:%f' %(total_pass_count/total_count))
    print('pass audio:%s' %pass_audio_list)