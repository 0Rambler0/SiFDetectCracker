import os
import sys
from utils.utils import Genotype
import torch
sys.path.append('../..')
from audio_processing import ReadAudio
from torch.utils.data import DataLoader, Dataset
import numpy as np
from models.model import Network
import argparse


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

class Args():
    def __init__(self):
        self.data = '/path/to/your/LA'
        self.layers = 8
        self.init_channels = 64
        self.gru_hsize = 1024
        self.gru_layers = 3
        self.sinc_scale = 'mel'
        self.sinc_kernel = 128
        self.batch_size = 64
        self.sr = 16000
        self.eval = 'e'
        self.arch = "Genotype(normal=[('dil_conv_5', 1), ('dil_conv_3', 0), ('dil_conv_5', 1), ('dil_conv_5', 2), ('std_conv_5', 2), ('skip_connect', 3), ('std_conv_5', 2), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3', 0), ('std_conv_3', 1), ('dil_conv_3', 0), ('dil_conv_3', 2), ('skip_connect', 0), ('dil_conv_5', 2), ('dil_conv_3', 0), ('avg_pool_3', 1)], reduce_concat=range(2, 6))"
        self.is_mask = False
        self.is_trainable = False

def get_model(model_path):
    args = Args()
    OUTPUT_CLASSES = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    genotype = eval(args.arch)
    model = Network(args.init_channels, args.layers, args, OUTPUT_CLASSES, genotype)
    model.drop_path_prob = 0.0
    model.load_state_dict(torch.load(os.path.join(model_path,'epoch_66.pth'), map_location=device))
    model.eval()
    model.cuda()

    return model 

def predict(audio_path, model):
    args = Args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    genotype = eval(args.arch)
    # print(model)
    # model = model.to(device=device)
    eval_set = EvalDataSet(audio_path)
    eval_set_loader = DataLoader(eval_set, batch_size=1, shuffle=False)
    score_list = [None]*eval_set.length
    min_score = 0
    pass_count = 0
    for audio, fname, samplerate in eval_set_loader:
        audio = audio.to(device=device, dtype=torch.float)
        output = model(audio)
        output = model.forward_classifier(output)
        score = (output[:, 1]  
                    ).data.cpu().numpy().ravel()[0]  
        value, predict = output.max(dim=1)
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
    parser = argparse.ArgumentParser('ASVSpoof2019 model')
    parser.add_argument('--data', type=str, default='/path/to/your/LA', 
                    help='location of the data corpus')   
    parser.add_argument('--layers', type=int, default=8)
    parser.add_argument('--init_channels', type=int, default=64)
    parser.add_argument('--gru_hsize', type=int, default=1024)
    parser.add_argument('--gru_layers', type=int, default=3)
    parser.add_argument('--sinc_scale', type=str, default='mel', help='the ytpe of sinc layer')
    parser.add_argument('--sinc_kernel', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_set', type=str, default=64)
    parser.add_argument('--silence_factor', type=str, default=1)
    parser.add_argument('--sr', type=int, default=16000, help='default sampling rate')
    parser.add_argument('--eval', type=str, default='e', help='to use eval or dev')
    parser.add_argument('--arch', type=str, default="Genotype(normal=[('dil_conv_5', 1), ('dil_conv_3', 0), ('dil_conv_5', 1), ('dil_conv_5', 2), ('std_conv_5', 2), ('skip_connect', 3), ('std_conv_5', 2), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3', 0), ('std_conv_3', 1), ('dil_conv_3', 0), ('dil_conv_3', 2), ('skip_connect', 0), ('dil_conv_5', 2), ('dil_conv_3', 0), ('avg_pool_3', 1)], reduce_concat=range(2, 6))", help='the searched architecture')

    parser.set_defaults(is_mask=False)
    parser.set_defaults(is_trainable=False)
    
    args = parser.parse_args()
    OUTPUT_CLASSES = 2

    # test_set = sys.argv[1]
    # silence_factor = sys.argv[2]

    model_path = '/home/dslab/hx/Vattack/test_model/raw-pc-darts-anti-spoofing/epoch_66.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    genotype = eval(args.arch)
    model = Network(args.init_channels, args.layers, args, OUTPUT_CLASSES, genotype)
    model.drop_path_prob = 0.0
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.cuda()
    # eval_set = EvalDataSet(path+'Nattack/audio/')
    audio_name_list = os.listdir('/home/dslab/hx/vfuzz/media/exp_data/Vattack_exp_2023_4_4/exp_long_audio/data/{}'.format(args.test_set))
    # audio_name_list = os.listdir('/home/dslab/hx/vfuzz/media/exp_data/Vattak_exp_2022_11_21/test_audio_set')
    is_logical = ('DF' == 'logical')
    total_pass_rate = 0
    total_count = len(audio_name_list)
    pass_audio_list = []
    for audio_name in audio_name_list:
        print('audio name:%s' %audio_name)
        eval_set = EvalDataSet('/home/dslab/hx/vfuzz/media/exp_data/Vattack_exp_2023_4_4/exp_long_audio/data/{}/{}/{}/'.format(args.test_set, audio_name, args.silence_factor))
        # eval_set = EvalDataSet('/home/dslab/hx/vfuzz/media/exp_data/Vattak_exp_2022_11_21/test_audio_set/')
        eval_set_loader = DataLoader(eval_set, batch_size=1, shuffle=False)
        pass_count = 0
        for audio, fname, samplerate in eval_set_loader:
            audio = audio.to(device=device, dtype=torch.float)
            output = model(audio)
            output = model.forward_classifier(output)
            score = (output[:, 1]  
                        ).data.cpu().numpy().ravel()[0]  
            value, predict = output.max(dim=1)
            if predict == 1:
                pass_count += 1
            print('fname:%s score:%f  label:%s' %(fname[0], score, predict.item()))
        pass_rate = pass_count/len(eval_set_loader)
        total_pass_rate += pass_rate
        print('pass rate:%f' %pass_rate)
        if pass_rate > 0.8:
            pass_audio_list.append(audio_name)
    print('total pass rate:%f average_pass_rate:%f' %(total_pass_rate, total_pass_rate/total_count))
    # print('total pass rate:%f' %(total_pass_count/total_count))
    print('pass audio:%s' %pass_audio_list)