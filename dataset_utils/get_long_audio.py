import librosa
import os 

if __name__ == '__main__':
    file_path = '/home/dslab/hx/vfuzz/media/data/LA/ASVspoof2019_LA_eval/flac'
    save_path = '/home/dslab/hx/vfuzz/media/data/LA/eval_long_audio'

    file_list = os.listdir(file_path)
    for filename in file_list:
        duration = librosa.get_duration(filename='{}/{}'.format(file_path,filename))
        if duration > 4:
            cmd = 'cp {}/{} {}/{}'.format(file_path, filename, save_path ,filename)
            os.system(cmd)
    os.chdir(save_path)
    file_list = os.listdir(save_path)
    for fname in file_list:
        fname = fname.split('.')[0]
        cmd2 = 'sox {}.flac -n noiseprof | sox {}.flac {}.wav noisered - 0.01'.format(fname, fname, fname)
        os.system(cmd2)
        print('processed file name:{}'.format(fname))