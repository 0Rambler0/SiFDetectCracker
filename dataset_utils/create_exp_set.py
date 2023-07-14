import os 

if __name__ == '__main__':
    protocol_path = '/home/dslab/hx/Vattack/dataset_utils/long_audio_protocol'
    save_path = '/home/dslab/hx/vfuzz/media/exp_data/Vattak_exp_2022_11_21/test_audio_set_new'
    long_audio_path = '/home/dslab/hx/vfuzz/media/data/LA/eval_long_audio'
    test_protocol_path = 'test_protocol'

    approach_list = ['A07','A08','A09','A10','A11','A12','A13','A14','A15','A16','A17','A18','A19']
    fname_dict = {}
    speaker_dict = {}

    protocol_f = open(protocol_path, 'r')
    test_protocol_f = open(test_protocol_path, 'w')
    protocol_lines = protocol_f.readlines()
    for line in protocol_lines:
        line = line.strip().split(' ')
        fname = line[0]
        approach = line[2]
        speaker = line[1]
        if approach == 'bonafide':
            continue
        if approach not in fname_dict.keys():
            fname_dict[approach] = [fname]
            speaker_dict[approach] = [speaker]
            test_protocol_f.write('{} {} {}\n'.format(fname,speaker,approach))
        else:
            if speaker not in speaker_dict[approach] and len(fname_dict[approach]) < 15:
                fname_dict[approach].append(fname)
                speaker_dict[approach].append(speaker)
                test_protocol_f.write('{} {} {}\n'.format(fname,speaker,approach))
    for approach in fname_dict:
        for fname in fname_dict[approach]:
            source_path = '{}/{}.wav'.format(long_audio_path, fname)
            target_path = '{}/{}.wav'.format(save_path, fname)
            cmd = 'cp {} {}'.format(source_path, target_path)
            os.system(cmd)