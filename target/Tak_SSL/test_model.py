import os 
import sys
sys.path.append('../..')
from audio_processing import ReadAudio
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
from model import Model

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

def get_model(model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:%s' %device)

    model = Model(device).to(device)
    model.load_state_dict(torch.load(os.path.join(model_path,'pre-trained—LA_model.pth'), 
                                     map_location=device))
    model = model.to(device=device)
    model.eval()

    return model

def predict(audio_path, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
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
        label = predict.item()
        if label == 1:
            pass_count += 1
        if score < min_score and label == 1:
            min_score = score
        score_list[int(fname[0].split('.')[0])-1] = score
    pass_rate = pass_count/len(eval_set_loader)

    return score_list, min_score, pass_rate