from tensorflow.keras import models
import cv2
import sys
import os
import numpy as np
import shutil

def load_model(path):
    model_name = os.path.join(path, "model_Deep4SNet.h5")
    weights_model = os.path.join(path, "weights_Deep4SNet.h5")
    model = models.load_model(model_name)
    model.load_weights(weights_model)

    return model

def load_img(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (150, 150))
    img = np.reshape(img, [1, 150, 150, 3])

    return img

def predict(img, model):
    p = [0, 0]
    classes = model.predict(img)
    # classes = model(img)
    p[0] = float(classes[0])
    p[1] = 1 - p[0]

    return p

def AudioConvert(read_path, save_path):
    lines=[]
    path = '/home/dslab/hx/vfuzz/media/hx_workspace/SiFDetectCracker/refactor_project/target/Deep4SNet/gen_pic_from_dir.m'
    f=open(path,'r') 
    for line in f:
        lines.append(line)
    f.close()
    lines[1] = 'read_path = ' + '\'' + read_path + '/\';'+'\n'
    lines[2] = 'save_path = ' + '\'' + save_path + '/\';'+'\n'
    s=''.join(lines)
    f=open(path,'w+') #重新写入文件
    f.write(s)
    f.close()
    os.chdir('/home/dslab/hx/vfuzz/media/hx_workspace/SiFDetectCracker/refactor_project/target/Deep4SNet')
    os.system('matlab -nosplash -nodesktop -r gen_pic_from_dir')
    os.chdir('/home/dslab/hx/vfuzz/media/hx_workspace/SiFDetectCracker/refactor_project')

def predict_pipeline(model_path, file_path, save_path):
    model = load_model(model_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print('start audio convert')
    # print(file_path)
    # print(save_path)
    AudioConvert(file_path, save_path)
    print('convert over')
    img_list = os.listdir(save_path)
    f_list = [None]*len(img_list)
    print('start scoring')
    count = 0
    pass_count = 0
    f_list = [None]*len(img_list)
    for img_name in img_list:
        count += 1
        img_code = int(img_name.split('.')[0]) - 1
        img = load_img(os.path.join(save_path, img_name))
        result = predict(img, model)
        if result[0] > 0.5:
            pass_count += 1
            f_list[img_code] = 1
        else:
            f_list[img_code] = 0
    pass_rate = pass_count/len(img_list)
    shutil.rmtree(save_path)

    return f_list, pass_rate

if __name__ == '__main__':
    # audio_path = '../audio/test2.wav'
    # print(os.path)
    path = '/home/dslab/hx/Vattack/'
    test_set = sys.argv[1]
    silence_factor = sys.argv[2]
    model = load_model(path+'/test_model/Deep4SNet/')
    audio_name_list = os.listdir('/home/dslab/hx/vfuzz/media/exp_data/Vattack_exp_2023_4_4/exp_long_audio/data/{}/'.format(test_set))
    total_pass_rate = 0
    total_count = len(audio_name_list)
    pass_audio_list = []
    for audio_name in audio_name_list:
        print('当前测试音频:%s' %audio_name)
        read_path = '/home/dslab/hx/vfuzz/media/exp_data/Vattack_exp_2023_4_4/exp_long_audio/data/{}/{}/{}/'.format(test_set, audio_name, silence_factor)
        save_path = path+'test_model/Deep4SNet/audio_img/test/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        AudioConvert(read_path, save_path)
        img_list = os.listdir(save_path)
        print('开始进行打分')
        pass_count = 0
        for img_name in img_list:
            img = load_img(save_path+img_name)
            result = predict(img, model)
            # print('测试结果 人声概率:%f 伪造概率：%f' %(result[0], result[1]))
            if result[0] > result[1]:
                pass_count += 1
        pass_rate = pass_count/len(img_list)
        total_pass_rate += pass_rate
        print('测试通过率:%f' %(pass_rate))
        shutil.rmtree(save_path)
        if pass_rate > 0.8:
            pass_audio_list.append(audio_name)
    print('total pass rate:%f average_pass_rate:%f' %(total_pass_rate, total_pass_rate/total_count))
    print('pass audio:%s' %pass_audio_list)
    # os.system('matlab -nojvm -r gen_pic_from_dir')
