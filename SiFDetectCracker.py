import shutil
import numpy as np
import concurrent.futures
from scipy.io import wavfile
import os 
import time

def sigmoid(x):
    s = 1 / (1 + np.exp(-0.5*x))
    return s

def Loss(score_list, min_score):
    """
    功能：计算样本的损失函数值
    输入：
        score_list：预测分值列表
        min_score:score_list中将音频判定为真的最小score
    输出：
        f_list：损失函数值列表
    """
    f_list = []

    for score in score_list:
        f = sigmoid(score)
        f_list.append(f)

    return f_list

def Clip(audio):
    for i in audio:
        if i > 1:
            i = 1
        if i < -1:
            i = -1
    
    return audio

class SiFDetectCracker():
    def __init__(self):
        self.lr = 0
        self.population = 0 
        self.init_mu = None
        self.mu = None
        self.sigma = None
        self.sigma_step_size = None
        self.t_step_size = None
        self.t_len = None
        self.threshold = 0
        self.threshold_n = 0
        self.threshold_t = 0
        self.perturbation_list = None
        self.fake_sample_name = None
        self.fake_sample = None
        self.perturbation_seed = None
        self.f_list = None
        self.z_score = None
        self.best_param = {}
    
    def set_param(self, lr, population, mu, sigma, sigma_step_size, t_step_size, t_len, 
                 threshold, threshold_n, threshold_t):
        self.lr = lr
        self.population = population 
        self.init_mu = mu
        self.mu = mu
        self.sigma = sigma
        self.sigma_step_size = sigma_step_size
        self.t_step_size = t_step_size
        self.t_len = t_len
        self.threshold = threshold
        self.threshold_n = threshold_n
        self.threshold_t = threshold_t
        self.best_param = {'pass_rate':0.0,
                           'mu':mu,
                           'sigma':sigma,
                           't_len':t_len}

    def set_fake_sample(self, fake_sample, sample_name, perturbation_seed):
        self.fake_sample = fake_sample
        self.fake_sample_name = sample_name
        self.perturbation_seed = perturbation_seed

    def param_summary(self, sr=16000):
        print('initial parameters:')
        print('  mu:{:.4f} sigma:{:.4f} t_len:{:.3f}'.format(np.mean(self.mu), 
                                                 self.sigma/np.std(self.perturbation_seed), 
                                                 self.t_len/len(self.fake_sample)))
        print('update parameters:')
        print('  sigma_step_size:{:.4f} t_step_size:{:.3f}'.format(self.sigma_step_size/np.std(self.perturbation_seed),
                                                           self.t_step_size/len(self.fake_sample)))
        print('threshold parameters:')
        print('threshold_n:{:.4f} threshold_t:{:.2f}s'.format(self.threshold_n/np.linalg.norm(self.fake_sample),
                                                     self.threshold_t/sr))


    def initialize_sample(self):
        """
        Add perturbation seed into sample 
        """
        sample = self.fake_sample
        perturbation = self.perturbation_seed
        sample_len = len(sample)
        if len(perturbation) >= sample_len:
            init_sample = sample + perturbation[0:sample_len]
            new_perturbation = perturbation[0:sample_len]
        else:
            new_perturbation = perturbation
            i = int(sample_len/len(perturbation)) - 1
            for j in range(0, i):
                new_perturbation = np.hstack((new_perturbation, perturbation))
            new_perturbation = np.hstack((new_perturbation, perturbation[0:sample_len-len(new_perturbation)]))
            init_sample = sample + new_perturbation
        self.fake_sample = init_sample

    def get_perturbation_samples(self, sr=16000):
        """
        Get perturbation seed Z from a normal distribution
        """
        size = (self.population, len(self.fake_sample)) 
        Z = np.random.normal(self.mu, self.sigma, size)
        calculate_pool = concurrent.futures.ProcessPoolExecutor(max_workers=15)
        num_Z = int(self.population/8)
        res_list = []
        print('start perturbation check')
        for i in range(8):
            tmp_Z = Z[i*num_Z:(i+1)*num_Z]
            res = calculate_pool.submit(self.check_perturbation, tmp_Z)
            res_list.append(res)
        calculate_pool.shutdown()
        new_Z = None
        for i in range(len(res_list)):
            new_Z = res_list[i].result() if i == 0 else np.vstack([new_Z, res_list[i].result()])
        Z = new_Z
        self.perturbation_list = Z
        print('check over')
          
    def check_perturbation(self, Z_list, sr=16000):
        """
        Check if the value of the perturbation exceeds the threshold
        """
        tmp_Z = []
        # for Z_i in perturbation_list:
        #     tmp_Z.append(BandstopFilter(Z_i, 1, 3000, sr))
        # perturbation_list = tmp_Z
        for Z_i in Z_list:
            for i in Z_i:
                if np.abs(i) > self.threshold:
                    if i >= 0:
                        i = self.threshold
                    else:
                        i = -self.threshold
        return Z_list
    
    def Z_score(self):
        """
        calculate z_score
        """
        z_score = []
        f_mean = np.mean(self.f_list)
        f_std = np.std(self.f_list)
        print('mean of loss:%f, standard deviation:%f' %(f_mean, f_std))
        if f_std == 0:
            print('warning, standard deviation is zero')
            return None
        for f_i in self.f_list:
            z_i = (f_i - f_mean)/f_std
            z_score.append(z_i)
        self.z_score = z_score
    
    def update_mu(self, lr):
        """
        Update mean of perturbation according to z_score
        """
        if len(self.perturbation_list) != len(self.z_score):
            print('The number of perturbation samples is not equal to the number of z_score')
            return 'error'
        modify = self.z_score[0]*self.perturbation_list[0]
        for i in range(1, len(self.perturbation_list)):
            modify += self.z_score[i]*self.perturbation_list[i]
        mu_new = self.mu - lr/(len(self.perturbation_list)*self.sigma)*modify
        self.mu = mu_new
        # print(mu_new)

    def add_time_perturbation(self, audio, noise_seed):
        """
        Add time perturbation
        Input:
            audio(1D-array): target audio
            noise_seed(1D-array): time perturbation sample
        Output
            new_audio(1D-array): audio after adding time perturbation
        """
        if type(self.t_len) != type(1):
            raise 't_len muse be an integer'
        if len(noise_seed) >= self.t_len:
            new_noise = noise_seed[0:self.t_len-1]
        else:
            new_noise = noise_seed
            i = int(self.t_len/len(noise_seed)) - 1
            for j in range(0, i):
                new_noise = np.hstack((new_noise, noise_seed))
            new_noise = np.hstack((new_noise, noise_seed[0:self.t_len-len(new_noise)]))
        new_audio = np.hstack((new_noise, audio))
        new_audio = np.hstack((new_audio, new_noise))

        return new_audio
    
    def worker(self, start_code, size, path, sr=16000, silence=None, mode='normal'):
        """
        Generate adversarial samples
        Input:
            start_code(int): start code of adversarial samples 
            size(int): range of selecting perturbation seed
            path(str): adversarial samples save path
            sr(int):sample rate
            silence(float): silence factor
            mode(str): perturbation mode ("normal", "no time", "no noise") 
        """
        code = start_code 
        wavfile.write('perturbation.wav', sr, (32768*self.perturbation_list[0]).astype(np.int16))
        end_code = start_code+size if start_code+size < len(self.perturbation_list) else len(self.perturbation_list)
        for delta in self.perturbation_list[start_code:end_code]:
            if silence != None:
                delta = silence*delta
            if mode == "normal":
                new_X = Clip(self.fake_sample+delta)
                new_X = self.add_time_perturbation(new_X, delta)
            elif mode == "no_time":
                new_X = Clip(self.fake_sample+delta)
            elif mode == "no_noise":
                new_X = self.add_time_perturbation(new_X, delta)
            else:
                raise "perturbation mode error, please check your code"
            wavfile.write(os.path.join(path,str(code)+'.wav'), sr, (32768*new_X).astype(np.int16))
            code += 1
        
        return

    def update_best_param(self, pass_rate, mode='normal'):
        if pass_rate > self.best_param['pass_rate']:
            self.best_param['pass_rate'] = pass_rate
            if mode == 'normal':
                self.best_param['mu'] = self.mu
                self.best_param['t_len'] = self.t_len
                self.best_param['sigma'] = self.sigma
                print('Update best param, mu:{:.6f} sigma:{:.6f} t_len:{:.6f}'.format(np.mean(self.mu), 
                                                                                    self.sigma/np.std(self.perturbation_seed), 
                                                                                    self.t_len/len(self.fake_sample)))
            elif mode == 'no_time':
                self.best_param['mu'] = self.mu
                self.best_param['sigma'] = self.sigma
                print('Update best param, mu:{:.6f} sigma:{:.6f}'.format(np.mean(self.mu), 
                                                                        self.sigma/np.std(self.perturbation_seed)))
            elif mode == 'no_noise':
                self.best_param['t_len'] = self.t_len
                print('Update best param, t_len:{:.6f}'.format(self.t_len/len(self.fake_sample)))
            else:
                raise "perturbation mode error, please check your code"
            
    def get_max_perturbation(self):
        max_perturbation = self.mu+2*self.sigma if np.linalg.norm(self.mu+2*self.sigma)>np.linalg.norm(self.mu-2*self.sigma) else self.mu-2*self.sigma

        return max_perturbation

    def param_search(self, max_iteration_num, tmp_save_path, 
                     model, target, target_name, 
                     sr=16000, mode='normal', is_label_only=False):
        iteration = 0
        error_count = 0 #the count of mu > threshold_n
        fail_count = 0 #the count of pass rate = 0
        lowrate_count = 0
        sigma_fail_count = 0
        total_generate_time = 0 #cost time without model prediction
        highrate_count = 0
        lr_tmp = self.lr

        print('target:{}'.format(target_name))
        self.param_summary()
        if not os.path.exists(tmp_save_path):
                os.mkdir(tmp_save_path)
        print('start parameter search')
        for iteration in range(1, max_iteration_num+1):
            print("{:*^50s}".format("new iteration"))
            print('iteration {} start'.format(iteration), flush=True)
            start_time = time.time()
            if mode != 'no_noise':
                print('perturbation generating ...')
                self.get_perturbation_samples()
                print('perturbation generate over')
                new_path = os.path.join(tmp_save_path, str(iteration))
                if not os.path.exists(new_path):
                    os.mkdir(new_path)
                print('generating adversarial samples ...')
                worker_pool = concurrent.futures.ProcessPoolExecutor(max_workers=20)
                start_pointer = 0
                for i in range(0, int(self.population/100) + 1):
                    if start_pointer >= self.population:
                        break
                    worker_pool.submit(self.worker,start_pointer,100,new_path,mode=mode)
                    start_pointer += 100
                worker_pool.shutdown()
                print('adversarial samples generate over')
                total_generate_time += time.time() - start_time
                tmp_time1 = time.time()
                print('start prediction')
                if target_name == 'Deep4SNet':
                    img_save_path = os.path.join(os.getcwd(), 'target/Deep4SNet/audio_img', str(iteration))
                    score_list, pass_rate = target(model, new_path, img_save_path)
                else:
                    score_list, min_score, pass_rate = target(new_path, model)
                print('prediction over')
                print('prediction time: {}'.format(time.time()-tmp_time1))
                update_start_time = time.time()
                self.f_list = score_list if is_label_only else Loss(score_list, min_score)
                # f_ad_list = LossAdjust(f_list, self.perturbation_list)
                self.Z_score()
                shutil.rmtree(new_path)
                print('pass rate of iteration {}:{:.1f}%'.format(iteration, pass_rate*100))
                if self.z_score == None:
                    fail_count = 2
                max_perturbation = self.get_max_perturbation()
                if pass_rate == 0.0:
                    if mode == 'normal':
                        if (sigma_fail_count == 2 and self.t_len <= self.threshold_t):
                            print('increase t_len')
                            self.t_len += self.t_step_size
                            print('new t_len:%f' %(self.t_len/len(self.fake_sample)))
                            sigma_fail_count = 0
                            end_time = time.time()
                            total_generate_time += end_time - update_start_time
                            print('time cost of iteration {}:{:.1f}s'.format(iteration, (end_time - start_time)))
                            continue
                    if fail_count == 2:
                        print('pass rate is zero, increase sigma')
                        self.sigma = self.sigma + self.sigma_step_size
                        self.mu = np.zeros(len(self.fake_sample))
                        print('new sigma:%f' %(self.sigma/np.std(self.perturbation_seed)))
                        fail_count = 0
                        sigma_fail_count += 1
                        end_time = time.time()
                        total_generate_time += end_time - update_start_time
                        print('time cost of iteration {}:{:.1f}s'.format(iteration, (end_time - start_time)))
                        continue
                    fail_count += 1
                    if np.linalg.norm(np.zeros(len(self.fake_sample))+2*self.sigma) > self.threshold_n:
                        print('Standard deviation exceeds threshold')
                        if mode == 'normal':
                            if self.t_len <= self.threshold_t:
                                self.sigma = self.sigma*0.5
                                print('Increase t_len')
                                self.t_len += self.t_step_size
                                print('new t_len:%f' %(self.t_len/len(self.fake_sample)))
                                sigma_fail_count = 0
                                end_time = time.time()
                                total_generate_time += end_time - update_start_time
                                print('time cost of iteration {}:{:.1f}s'.format(iteration, (end_time - start_time)))
                                continue
                            else:
                                self.mu = self.best_param['mu']
                                self.sigma = 0.5*self.sigma
                                self.t_len = self.best_param['t_len']
                                lr_tmp = 0.8*lr_tmp
                                total_generate_time += end_time - update_start_time
                                print('Standard deviation and t_len exceeds threshold, decrease lr')
                                print('New learning rate:{:.4f}'.format(lr_tmp))
                                continue
                        else:
                            self.mu = self.best_param['mu']
                            self.sigma = 0.5*self.sigma
                            lr_tmp = 0.8*lr_tmp
                            total_generate_time += end_time - update_start_time
                            print('Standard deviation and t_len exceeds threshold, decrease lr')
                            print('New learning rate:{:.4f}'.format(lr_tmp))
                            continue
                if np.linalg.norm(max_perturbation) > self.threshold_n and pass_rate > 0.7: #mu的均值超过门限且本轮通过率大于70%时修改error_count
                    error_count += 1
                    if error_count > 5:
                        print('mu too large, increase sigma')
                        self.sigma += self.sigma_step_size
                        self.mu = np.zeros(len(self.fake_sample))
                        print('new sigma:%f' %(self.sigma/np.std(self.perturbation_seed)))
                        error_count = 0
                        sigma_fail_count += 1
                        end_time = time.time()
                        total_generate_time += end_time - update_start_time
                        print('time cost of iteration {}:{:.1f}s'.format(iteration, (end_time - start_time)))
                        continue
                    else:
                        print('mu too large, error count + 1')
                        print('max_perturbation:{:.4f}'.format(np.linalg.norm(max_perturbation)/np.linalg.norm(self.fake_sample)))
                if pass_rate > self.best_param['pass_rate'] \
                and np.linalg.norm(max_perturbation) <= self.threshold_n:
                    self.update_best_param(pass_rate, mode=mode)
                if pass_rate > 0.8 and np.linalg.norm(max_perturbation) <= self.threshold_n:
                    highrate_count += 1
                    if highrate_count >= 6:
                        print('The pass rate exceeds 0.8 many times, search ends')
                        total_generate_time += time.time() - update_start_time
                        break
                if pass_rate > 0.9:
                    if np.linalg.norm(max_perturbation) <= self.threshold_n:
                        print('The pass rate meets the requirements')
                        end_time = time.time()
                        total_generate_time += end_time - update_start_time
                        print('time cost of iteration {}:{:.1f}s'.format(iteration, (end_time-start_time)))
                        break
                    else:
                        print('The pass rate meets the requirements, but mu exceeds threshold')
                        self.mu = 0.5*self.mu
                        sigma_fail_count += 1
                        end_time = time.time()
                        total_generate_time += end_time - update_start_time
                        print('time cost of iteration {}:{:.1f}s'.format(iteration, (end_time-start_time)))
                        continue
                lr_tmp = 0.001 if pass_rate > 0.8 else lr_tmp
                self.update_mu(lr_tmp)
                print('new mu:{:.4f}, standard deviation of mu:{:.4f}'.format(np.mean(self.mu), np.std(self.mu)))
                end_time = time.time()
                total_generate_time += end_time - update_start_time
                print('time cost of iteration {}:{:.1f}s'.format(iteration, (end_time-start_time)))
            else:
                self.get_perturbation_samples()
                new_path = os.path.join(tmp_save_path, str(iteration))
                if not os.path.exists(new_path):
                    os.mkdir(new_path)
                print('generating adversarial samples ...')
                worker_pool = concurrent.futures.ProcessPoolExecutor(max_workers=20)
                start_pointer = 0
                for i in range(0, int(self.population/100) + 1):
                    if start_pointer >= self.population:
                        break
                    worker_pool.submit(self.worker,start_pointer,100,new_path,mode=mode)
                    start_pointer += 100
                worker_pool.shutdown()
                print('adversarial samples generate over')
                total_generate_time += time.time() - start_time
                total_generate_time += time.time() - start_time
                print('start prediction')
                if target_name == 'Deep4SNet':
                    img_save_path = os.path.join(os.getcwd(), 'target/Deep4SNet/audio_img', str(iteration))
                    score_list, pass_rate = target(new_path, img_save_path)
                else:
                    score_list, min_score, pass_rate = target(new_path, model)
                print('prediction over')
                update_start_time = time.time()
                self.f_list = score_list if target=='Deep4SNet' else Loss(score_list, min_score)
                # f_ad_list = LossAdjust(f_list, self.perturbation_list)
                self.Z_score()
                shutil.rmtree(new_path)
                print('pass rate of iteration {}:{:.1f}%'.format(iteration, pass_rate*100))
                # if pass_rate > self.best_param['pass rate']:
                self.update_best_param(pass_rate, mode=mode)
                if pass_rate > 0.9 and self.t_len <= self.threshold_t:
                    print('The pass rate meets the requirements')
                    break
                else:
                    if self.t_len <= self.threshold_t:
                        print('Increase t_len')
                        self.t_len += self.t_step_size
                        print('new t_len:%f' %(self.t_len/len(self.fake_sample)))
                        sigma_fail_count = 0
                        end_time = time.time()
                        print('time cost of iteration {}:{:.1f}s'.format(iteration, (end_time-start_time)))
                        continue
                    else:
                        print('t_len too large, search failed')
                        break
        print('Parameters search over')
        avg_time = total_generate_time/iteration

        return iteration, avg_time
    
    def evaluate(self, population, save_path, model, target, target_name, mode):
        pass_rate_list = []
        save_path = os.path.join(save_path, mode, 'data')
        print('Start evaluation')
        if mode == 'normal':
            self.mu = self.best_param['mu']
            self.t_len = self.best_param['t_len']
            self.sigma = self.best_param['sigma']
            print('test parameters  mu:%f t_len:%f sigma:%f' %(np.mean(self.mu), 
                                                               self.t_len/len(self.fake_sample), 
                                                               self.sigma/np.std(self.perturbation_seed)))
            print('perturbation generating ...')
            self.get_perturbation_samples()
            print('perturbation generate over')
        if mode == 'no_time':
            self.mu = self.best_param['mu']
            self.sigma = self.best_param['sigma']
            print('test parameters  mu:%f sigma:%f' %(np.mean(self.mu), 
                                                      self.sigma/np.std(self.perturbation_seed)))
            print('perturbation generating ...')
            self.get_perturbation_samples()
            print('perturbation generate over')
        if mode == 'no_noise':
            self.t_len = self.best_param['t_len']
            print('test parameters  t_len:%f' %(self.t_len/len(self.fake_sample)))
        save_path = os.path.join(save_path, target_name, self.fake_sample_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        silence_factor_list = [1, 0.5, 0.2, 0.05]
        for silence_factor in silence_factor_list:
            new_path = os.path.join(save_path, str(silence_factor))
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            worker_pool = concurrent.futures.ProcessPoolExecutor(max_workers=20)
            print('generating adversarial samples ...')
            start_pointer = 0
            for i in range(0, int(population/100) + 1):
                if start_pointer >= self.population:
                    break
                worker_pool.submit(self.worker,start_pointer,100,new_path,silence=silence_factor,mode=mode)
                start_pointer += 100
            worker_pool.shutdown()
            print('adversarial samples generate over')
            if target_name == 'Deep4SNet':
                img_save_path = os.path.join(os.getcwd(), 'target/Deep4SNet/audio_img', 'eval')
                score_list, pass_rate = target(model, new_path, img_save_path)
            else:
                score_list, min_score, pass_rate = target(new_path, model)
            pass_rate_list.append(pass_rate)
            print('silence factor:%f    pass rate:%f%%' %(silence_factor, pass_rate*100))

        return pass_rate_list


        
        

                    






                



