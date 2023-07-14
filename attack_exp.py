from SiFDetectCracker import SiFDetectCracker
import argparse
import sys
from audio_processing import ReadAudio, GetSiFeatures
import os 
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SiFDetectCracker')
    parser.add_argument('--target', type=str, default='Deep4SNet', help='change target you want to attack')
    parser.add_argument('--test_set', type=str, default='/data_hdd/lx20/workspaces/hx_workspace/data/Vattack_exp/exp_2023_4_12/test_audio_set_new', help='select test_set')
    parser.add_argument('--save_path', type=str, default='/data_hdd/lx20/workspaces/hx_workspace/SiFDetectCracker/exp_data/SiFDetectCracker_exp')
    parser.add_argument('--tmp_path', type=str, default='/data_hdd/lx20/workspaces/hx_workspace/SiFDetectCracker/data/tmp', help='index of save template adversarial samples')
    parser.add_argument('--perturbation_seed', type=str, default='/data_hdd/lx20/workspaces/hx_workspace/SiFDetectCracker/refactor_project/file1486.wav_16k.wav_norm.wav_mono.wav_silence.wav')
    parser.add_argument('--target_path', type=str, default='/data_hdd/lx20/workspaces/hx_workspace/SiFDetectCracker/refactor_project/target', help='index of all targets')
    #initial parameters set
    parser.add_argument('--lr', type=float, default='0.01')
    parser.add_argument('--sigma', type=float, default='0.1')
    parser.add_argument('--iteration_num', type=int, default='100')
    parser.add_argument('--mode', type=str, default='normal', help='attack mode you want to select')
    args = parser.parse_args()

    test_set_list = os.listdir(args.test_set)
    population = 400 
    eval_population = 200
    lr = args.lr
    num_iteration = args.iteration_num 
    tmp_save_path = args.tmp_path
    target_name = args.target
    model_path = os.path.join(args.target_path, target_name)
    mode = args.mode
    if target_name == 'Deep4SNet':
        sys.path.append('/data_hdd/lx20/workspaces/hx_workspace/SiFDetectCracker/refactor_project/target/Deep4SNet/')
        from test_model import predict_pipeline as predict_deep4snet
        target = predict_deep4snet
    elif target_name == 'Rawnet2':
        sys.path.append('/data_hdd/lx20/workspaces/hx_workspace/SiFDetectCracker/refactor_project/target/Rawnet2/')
        from test_model import predict as predict_rawnet2
        target = predict_rawnet2
    elif target_name == 'RawGAT_ST':
        sys.path.append('/data_hdd/lx20/workspaces/hx_workspace/SiFDetectCracker/refactor_project/target/RawGAT_ST/')
        from test_model import predict as predict_rawgat
        target = predict_rawgat
    elif target_name == 'Raw_pc_darts':
        sys.path.append('/data_hdd/lx20/workspaces/hx_workspace/SiFDetectCracker/refactor_project/target/Raw_pc_darts/')
        from test_model import predict as predict_rawpcdarts
        target = predict_rawpcdarts
    else:
        raise 'unknown target'
    attacker = SiFDetectCracker()
    
    pass_rate_dict = {}
    avg_time_dict = {}
    perturbation_seed, front_noise, rear_noise = GetSiFeatures(args.perturbation_seed)
    if not os.path.exists(os.path.join(args.save_path,mode,'csv_res')):
            os.mkdir(os.path.join(args.save_path,mode,'csv_res'))
    if not os.path.exists(os.path.join(args.save_path,mode,'data',target_name)):
            os.makedirs(os.path.join(args.save_path,mode,'data',target_name))
    csvfile = open(os.path.join(args.save_path,mode,'csv_res','{}_res.csv'.format(target_name)), 'w', newline='')

    csvfile.write('audio,no silence,silence-0.5,silence-0.2,silence-0.05,iteration,avg-time\n')
    test_set_list = test_set_list
    for fname in test_set_list:
        print('target sample:{}'.format(fname.split('.')[0]))
        new_audio, sr = ReadAudio(os.path.join(args.test_set,fname))
        mu = np.zeros(len(new_audio))
        sigma = args.sigma*np.std(perturbation_seed)
        sigma_step_size = 0.01*np.std(perturbation_seed)
        t_step_size = int(0.1*len(new_audio)) if target_name=='Deep4SNet' else int(max(0.01*len(new_audio), 0.2*sr))
        threshold = np.abs(0.05*np.mean(new_audio)) 
        threshold_n = 0.07*np.linalg.norm(new_audio)
        t_len = int(0.05*len(new_audio)) if target_name=='Deep4SNet' else int(max(0.02*len(new_audio), sr*0.5))
        t_len = int(t_len)
        threshold_t = 2*sr
        attacker.set_param(lr, population, mu, sigma, sigma_step_size, t_step_size, t_len, 
                           threshold, threshold_n, threshold_t)
        attacker.set_fake_sample(new_audio, fname.split('.')[0], perturbation_seed)
        if target_name == 'Deep4SNet':
            attacker.initialize_sample()
        iteration, avg_time_dict[fname] = attacker.param_search(num_iteration, tmp_save_path, model_path, target, target_name, mode=mode)
        pass_rate_dict[fname] = attacker.evaluate(eval_population, args.save_path, model_path, target, target_name, mode)
        pass_rate_dict[fname].append(iteration)
        print("\n{:=^50s}\n".format("Split Line"))
    print ("{:<8} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}".format('audio','no silence','silence-0.5','silence-0.2','silence-0.05', 'interation', 'avg-time'))
    sum_list = [0, 0, 0, 0, 0, 0]
    for key in pass_rate_dict:
        print ("{:<8} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}".format(key.split()[0],
                                                          pass_rate_dict[key][0],
                                                          pass_rate_dict[key][1],
                                                          pass_rate_dict[key][2],
                                                          pass_rate_dict[key][3],
                                                          pass_rate_dict[key][4],
                                                          avg_time_dict[key]))
        csvfile.write('{},{},{},{},{},{},{}\n'.format(key.split()[0], 
                                                    pass_rate_dict[key][0], 
                                                    pass_rate_dict[key][1], 
                                                    pass_rate_dict[key][2], 
                                                    pass_rate_dict[key][3],
                                                    pass_rate_dict[key][4],
                                                    avg_time_dict[key]))
        for i in range(5):
            sum_list[i] += pass_rate_dict[key][i]
    average_list = []
    for i in range(5):
        average_list.append(sum_list[i]/len(test_set_list))
    print("{:<8} {:<12} {:<12} {:<12} {:<12}".format('average', average_list[0], average_list[1], average_list[2], average_list[3]))
    csvfile.write('average,{},{},{},{},{}'.format(average_list[0], 
                                                  average_list[1], 
                                                  average_list[2], 
                                                  average_list[3], 
                                                  average_list[4]))
    csvfile.close()

