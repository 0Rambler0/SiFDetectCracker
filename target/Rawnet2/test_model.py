from fileinput import filename
import torch
from model import RawNet
import os 
import yaml
import sys
sys.path.append('../..')
from audio_processing import ReadAudio
from torch.utils.data import DataLoader, Dataset
import numpy as np
from data_utils import *
from torchvision import transforms
from torch import Tensor, threshold


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
        audio, samplerate = ReadAudio(self.path + self.file_list[idx])
        audio = pad(audio)
        labels = 0
        
        return audio, labels, samplerate

def predict(audio_path, model_path):
    dir_yaml = os.path.join(model_path, 'model_config_RawNet.yaml')
    with open(dir_yaml, 'r') as f_yaml:
            parser1 = yaml.safe_load(f_yaml)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RawNet(parser1['model'], device).to(device)
    model.load_state_dict(torch.load(os.path.join(model_path,'pre_trained_DF_RawNet2.pth'), map_location=device))
    # print(model)
    model.eval()
    list_ids = os.listdir(audio_path)
    score_list = [None]*len(list_ids)
    eval_set = Dataset_ASVspoof2021_eval(list_IDs=list_ids, base_dir=audio_path)
    eval_set_loader = DataLoader(eval_set, batch_size=1, shuffle=False, drop_last=False)
    min_score = 0
    pass_count = 0
    threshold = -0.0048
    for audio, audio_name in eval_set_loader:
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
        score_list[int(audio_name[0].split('.')[0])-1] = score
    pass_rate = pass_count/len(eval_set_loader)

    return score_list, min_score, pass_rate

if __name__ == '__main__':
    path = '/home/dslab/hx/Vattack/'
    test_set = sys.argv[1]
    silence_factor = sys.argv[2]

    dir_yaml = path + 'test_model/Baseline_RawNet2/model_config_RawNet.yaml'
    with open(dir_yaml, 'r') as f_yaml:
            parser1 = yaml.safe_load(f_yaml)

    model_path = path + 'test_model/Baseline_RawNet2/pre_trained_DF_RawNet2.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RawNet(parser1['model'], device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    # print(model)
    model.eval()
    audio_name_list = os.listdir('/home/dslab/hx/vfuzz/media/exp_data/Vattack_exp_2023_4_4/exp_long_audio/data/{}'.format(test_set))
    total_pass_rate = 0
    total_count = len(audio_name_list)
    pass_audio_list = []
    for audio_name in audio_name_list:
        print('当前测试音频:%s' %audio_name)
        list_ids = os.listdir('/home/dslab/hx/vfuzz/media/exp_data/Vattack_exp_2023_4_4/exp_long_audio/data/{}/{}/{}/'.format(test_set, audio_name, silence_factor))
        eval_set = Dataset_ASVspoof2021_eval(list_IDs=list_ids, base_dir='/home/dslab/hx/vfuzz/media/exp_data/Vattack_exp_2023_4_4/exp_long_audio/data/{}/{}/{}/'.format(test_set, audio_name, silence_factor))
        is_logical = ('DF' == 'logical')
        """transforms = transforms.Compose([
            lambda x: pad(x),
            lambda x: Tensor(x)
        ])"""
        eval_set_loader = DataLoader(eval_set, batch_size=1, shuffle=False, drop_last=False)
        max_score = 0
        error_count = 0
        pass_count = 0
        threshold = -0.0048
        for audio, file_name in eval_set_loader:
            audio = audio.to(device=device, dtype=torch.float)
            output = model(audio)
            batch_score = (output[:, 1]
                        ).data.cpu().numpy().ravel().tolist()
            _, batch_pred = output.max(dim=1)
            batch_score = batch_score[0]
            labels = ['fake', 'real']
            label = batch_pred.item()
            if label == 0:
                error_count += 1
            else:
                pass_count += 1
            if -batch_score > max_score and label == 1:
                max_score = batch_score
            # print('audio name:%s   score:%f   label:%s' %(file_name[0], batch_score, labels[label]))
        pass_rate = pass_count/len(eval_set_loader)
        total_pass_rate += pass_rate
        # print('max score is:%f' %max_score)
        print('pass rate:%f' %pass_rate)
        if pass_rate >= 0.8:
            pass_audio_list.append(audio_name)
    print('total pass rate:%f average_pass_rate:%f' %(total_pass_rate, total_pass_rate/total_count))
    # print('total pass rate:%f' %(total_pass_count/total_count))
    print("pass audio:%s" %pass_audio_list)