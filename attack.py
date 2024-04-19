from SiFDetectCracker import SiFDetectCracker
import argparse
import sys
from audio_processing import ReadAudio, GetSiFeatures
import os 
import numpy as np
import json
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SiFDetectCracker')
    parser.add_argument('--config', type=str, default='./config/config.conf', help='select attack config file')
    args = parser.parse_args()
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())

    target_config = config["target_config"]
    path_config = config["path_config"]
    attack_config = config["attack_config"]

    test_set_list = os.listdir(path_config["data_path"])
    population = attack_config["population"] 
    eval_population = attack_config["eval_population"]
    lr = attack_config["lr"]
    num_iteration = attack_config["epoch_num"] 
    tmp_save_path = os.path.join(path_config["save_path"], "tmp")
    target_name = target_config["target_name"]
    model_path = os.path.join(path_config["target_path"], target_name)
    mode = attack_config["mode"]
    is_label_only = attack_config["label_only"]

    if os.path.exists(model_path):
        sys.path.append(model_path)
        from test_model import predict, get_model
        target = predict
        model = get_model(model_path)
    else:
         raise "target not exist"

    attacker = SiFDetectCracker()
    
    pass_rate_dict = {}
    avg_time_dict = {}
    perturbation_seed, front_noise, rear_noise = GetSiFeatures(attack_config["perturbation_seed"])
    if not os.path.exists(os.path.join(path_config["save_path"],mode,'csv_res')):
        os.makedirs(os.path.join(path_config["save_path"],mode,'csv_res'))
    if not os.path.exists(os.path.join(path_config["save_path"],mode,'data',target_name)):
        os.makedirs(os.path.join(path_config["save_path"],mode,'data',target_name))
    if not os.path.exists(tmp_save_path):
         os.makedirs(tmp_save_path)
    csvfile = open(os.path.join(path_config["save_path"],mode,'csv_res','{}_res.csv'.format(target_name)), 'w', newline='')

    csvfile.write('audio,no silence,silence-0.5,silence-0.2,silence-0.05,iteration,avg-time\n')
    csvfile.flush()
    test_set_list = test_set_list
    for fname in test_set_list:
        print('target sample:{}'.format(fname.split('.')[0]))
        new_audio, sr = ReadAudio(os.path.join(path_config["data_path"],fname))
        mu = np.zeros(len(new_audio))
        sigma = attack_config["sigma"]*np.std(perturbation_seed)
        sigma_step_size = attack_config["sigma_step_size"]*np.std(perturbation_seed)
        t_step_size = int(max(attack_config["t_step_size"][0]*len(new_audio), attack_config["t_step_size"][1]*sr))
        threshold = np.abs(attack_config["threshold"]*np.mean(new_audio)) 
        threshold_n = attack_config["threshold_n"]*np.linalg.norm(new_audio)
        t_len = int(max(attack_config["t_len"][0]*len(new_audio), sr*attack_config["t_len"][1]))
        threshold_t = attack_config["threshold_t"]*sr
        attacker.set_param(lr, population, mu, sigma, sigma_step_size, t_step_size, t_len, 
                           threshold, threshold_n, threshold_t)
        attacker.set_fake_sample(new_audio, fname.split('.')[0], perturbation_seed)
        if attack_config["is_initial"]:
            attacker.initialize_sample()
        iteration, avg_time_dict[fname] = attacker.param_search(num_iteration, tmp_save_path, model, target, target_name, mode=mode, is_label_only=is_label_only)
        pass_rate_dict[fname] = attacker.evaluate(eval_population, path_config["save_path"], model, target, target_name, mode)
        pass_rate_dict[fname].append(iteration)
        csvfile.write('{},{},{},{},{},{},{}\n'.format(fname.split('.')[0], 
                                                    pass_rate_dict[fname][0], 
                                                    pass_rate_dict[fname][1], 
                                                    pass_rate_dict[fname][2], 
                                                    pass_rate_dict[fname][3],
                                                    pass_rate_dict[fname][4],
                                                    avg_time_dict[fname]))
        csvfile.flush()
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

