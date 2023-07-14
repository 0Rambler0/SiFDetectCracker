from scipy.io import wavfile
import soundfile
import math
import pylab
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import librosa

def ReadAudio(path, sr=16000):
    data = librosa.load(path, sr=sr)[0]

    return data, sr

def enframe(data,win,inc):
    '''
    Framing processing of audio data
    input:
        data(1D-array):input audio data
        wlen(int):window size
        inc(int):moving step size of window 
    output:
        f(2D-array):data after framing
    '''
    nx = len(data) 
    try:
        nwin = len(win)
    except Exception as err:
        nwin = 1	
    if nwin == 1:
        wlen = win
    else:
        wlen = nwin
    nf = int(np.fix((nx - wlen) / inc) + 1)
    f = np.zeros((nf,wlen))  
    indf = [inc * j for j in range(nf)]
    indf = (np.mat(indf)).T
    inds = np.mat(range(wlen))
    indf_tile = np.tile(indf,wlen)
    inds_tile = np.tile(inds,(nf,1))
    mix_tile = indf_tile + inds_tile
    f = np.zeros((nf,wlen))
    for i in range(nf):
        for j in range(wlen):
            f[i,j] = data[mix_tile[i,j]]
    return f

def point_check(wavedata,win,inc):
    '''
    Speech signal start point and end point detection
    输入:
        wavedata(1D-array)：raw audio data
        win(int):window size
        inc(int):moving step size of window 
    输出:
        StartPoint(int):start point
        EndPoint(int):end point
    '''
    #1.Calculate the short-term zero-crossing rate
    FrameTemp1 = enframe(wavedata[0:-1],win,inc)
    FrameTemp2 = enframe(wavedata[1:],win,inc)
    signs = np.sign(np.multiply(FrameTemp1,FrameTemp2)) 
    signs = list(map(lambda x:[[i,0] [i>0] for i in x],signs))
    signs = list(map(lambda x:[[i,1] [i<0] for i in x], signs))
    diffs = np.sign(abs(FrameTemp1 - FrameTemp2)-0.01)
    diffs = list(map(lambda x:[[i,0] [i<0] for i in x], diffs))
    zcr = list((np.multiply(signs, diffs)).sum(axis = 1))

    #2.Calculate short-term energy
    amp = list((abs(enframe(wavedata,win,inc))).sum(axis = 1))
    
    # Set threshold
    print('Set threshold')
    ZcrLow = max([round(np.mean(zcr)*0.1),3])
    ZcrHigh = max([round(max(zcr)*0.1),5])
    AmpLow = min([min(amp)*15,np.mean(amp)*0.2,max(amp)*0.1])
    AmpHigh = max([min(amp)*15,np.mean(amp)*0.2,max(amp)*0.1])

    # Point detection
    MaxSilence = 0.02*len(zcr)
    MinAudio = 16 
    Status = 0 
    HoldTime = 0 
    SilenceTime = 0 
    print('Start point check')
    StartPoint = 0
    for n in range(len(zcr)):
        if Status ==0 or Status == 1:
            if amp[n] > AmpHigh or zcr[n] > ZcrHigh:
                StartPoint = n - HoldTime
                Status = 2
                HoldTime = HoldTime + 1
                SilenceTime = 0
            elif amp[n] > AmpLow or zcr[n] > ZcrLow:
                Status = 1
                HoldTime = HoldTime + 1
            else:
                Status = 0
                HoldTime = 0
        elif Status == 2:
            if amp[n] > AmpLow or zcr[n] > ZcrLow:
                HoldTime = HoldTime + 1
                SilenceTime = int(SilenceTime/2)
            else:
                SilenceTime = SilenceTime + 1
                if SilenceTime < MaxSilence:
                    HoldTime = HoldTime + 1
                elif (HoldTime - SilenceTime) < MinAudio:
                    Status = 0
                    HoldTime = 0
                    SilenceTime = 0
                else:
                    Status = 3
        elif Status == 3:
            break
        if Status == 3:
            break
    HoldTime = HoldTime - SilenceTime
    EndPoint = StartPoint + HoldTime
    StartPoint = win + StartPoint*inc
    EndPoint = win + EndPoint*inc

    return StartPoint,EndPoint,FrameTemp1

def PlotSpectrum(data, samplerate):
    fft_size = len(data)
    fft_data = np.fft.rfft(data)/fft_size
    freqs = np.linspace(0, samplerate/2, int(fft_size/2)+1)
    plt.plot(freqs, np.abs(fft_data))
    pylab.xlabel("frequence(Hz)")

def BandstopFilter(data, lowfrequency, highfrequency, samplerate):
    """
    BandstopFilter
    Input:
        data(1D-array):input data
        lowfrequency:low frequency threshold
        highfrequency:high frequency threshold
        samplerate:sample rate of data 
    Output:
        filted_data:data after filtering
    """
    w1 = 2*lowfrequency/samplerate
    w2 = 2*highfrequency/samplerate
    b, a = signal.butter(2, [w1, w2], 'bandstop')  
    filted_data = signal.filtfilt(b, a, data)

    return filted_data

def GetSiFeatures(path):
    """
    SiFs(Speaker-irrelative Features) extractor
    input:
        path(str):audio file path
    output:
        filted_audio(1D-array):Speech part after silence
        front_noise(1D-arrray):mute before speaker's voice 
        rear_noise(1D-array):mute after speaker's voice 
    """
    win = 128
    inc = 64

    audio, samplerate = ReadAudio(path)
    StartPoint,EndPoint,FrameTemp = point_check(audio,win,inc)
    cut_audio = audio[StartPoint:EndPoint]
    front_noise = audio[0:StartPoint]
    rear_noise = audio[EndPoint:]
    filted_audio = BandstopFilter(cut_audio, 100, 3000, samplerate)
    filted_audio = filted_audio*0.2

    return filted_audio, front_noise, rear_noise

def VoiceJoint(front_noise, audio, rear_noise):
    """
    Add mute into audio
    Input:
        front_noise: mute before voice 
        audio: target audio
        rear_noise: mute after voice 
    Output:
        new_audio: audio adding mute 
    """
    new_audio = np.hstack([front_noise, audio])
    new_audio = np.hstack([new_audio, rear_noise])

    return new_audio

def LenTransformation(audio, length):
    """
    Resize audio to specified length
    Input:
        audio(1D-array): target audio
        length(int): target length
    Output:
        new_audio(1D-array): audio after resize
    """
    if len(audio) >= length:
        new_audio = audio[0:length]
    else:
        new_audio = audio
        i = int(length/len(audio)) - 1
        for j in range(0, i):
            new_audio = np.hstack((new_audio, audio))
        new_perturbation = np.hstack((new_audio, audio[0:length-len(new_perturbation)]))

    return new_audio

if __name__ == "__main__":
    #定义音频数据采样时的窗口大小以及以及窗口滑动长度
    win = 128
    inc = 64

    audio, samplerate = ReadAudio('audio/test.wav')
    real_audio, samplerate_real = ReadAudio('audio/test2.wav')
    StartPoint,EndPoint,FrameTemp = point_check(real_audio,win,inc)
    cut_audio = real_audio[StartPoint:EndPoint]
    front_noise = real_audio[0:StartPoint]
    rear_noise = real_audio[EndPoint:]
    filted_audio = BandstopFilter(cut_audio, 1, 3000, samplerate_real)
    # soundfile.write('audio/filted_audio.wav', filted_audio, samplerate_real)
    soundfile.write('test3.wav', real_audio/5, samplerate_real)