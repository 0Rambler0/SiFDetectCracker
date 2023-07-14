import os 

if __name__ == '__main__':
    protocol_path = '/data_hdd/lx20/hx_workspace/data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
    save_path = 'long_audio_protocol'
    long_audio_path = '/data_hdd/lx20/hx_workspace/data/LA/eval_long_audio'
    
    protocol_f = open(protocol_path, 'r')
    save_f = open(save_path, 'w')
    protocol_lines = protocol_f.readlines()
    long_audio_list = os.listdir(long_audio_path)
    count = 0
    for line in protocol_lines:
        line = line.strip().split(' ')
        fname = line[1]
        approach = 'bonafide' if line[3]=='-' else line[3]
        speaker = line[0]
        if '{}.wav'.format(fname) in long_audio_list:
            save_f.write('{} {} {}\n'.format(fname, speaker, approach))
            count += 1
            print('processing: {}/{}'.format(count, len(long_audio_list)))