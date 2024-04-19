SiFDetectCracker: An Adversarial Attack Against Fake Voice Detection Based on Speaker-Irrelative Features
===============
This repository contains our implementation of the paper published in the ACM MultiMedia 2023, "SiFDetectCracker: An Adversarial Attack Against Fake Voice Detection Based on Speaker-Irrelative Features". 

[Paper link here](https://dl.acm.org/doi/abs/10.1145/3581783.3613841)

## Installation
First, clone the repository locally
```
$ git clone https://github.com/0Rambler0/SiFDetectCracker.git
```
Then, install all of the dependencies your target required. The dependencies of the target used in our experiments is shown in the README.md in the target index.

Deep4SNet use matlab to convert audio to image so you should install matlab if you want to attack Deep4SNet.

## Experiments

### Dataset
ASVspoof 2019 dataset is used in our experiment, which can can be downloaded from [here](https://datashare.is.ed.ac.uk/handle/10283/3336). We select 195 samples as the test samples by using the utils in dataset utils.

### Create evaluation set

To get the same test samples we used in experiment, please change the path in codes and run:

```
python3 dataset_utils/create_long_audio_protocol.py
python3 dataset_utils/get_long_audio.py
python3 dataset_utils/create_exp_set.py
```

### Evaluation

#### Edit config file
if you want to use the config file we provide, you must change the relative path in config file before your experiments. The parameter "mode" include "normal", "no_time" and "no_noise", which correspond to the experiments mode in our paper. For "no_noise" mode, please set "epoch_num" to 8.

#### Start:
```
python3 attack.py --config ..config/<target_name.conf>
```

### Add new target
Simply by adding a configuration file and a interface file, one can attack a new target.

To add a new target:
```
1.Create a new index at ./target/
2.Define a interface file named "test_model.py"
  - "test_model.py" must provide a model_inference function and a model create function.
  - Please refer to the existing "test_model.py" we provide at ./target/RawNet2/.
3.Create a new configuration file
  - Please refer to the existing configuration file we provide
4.run python3 attack.py--config ..config/<your_config.conf>
```

### License
```
Copyright (c) 2021-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
## Contact
For any query regarding this repository, please contact:
- Xuan Hai: haix21@lzu.edu.cn
- Xin Liu: bird@lzu.edu.cn

## Citation
If you use this code in your research please use the following citation:
```bibtex

@inproceedings{hai2023sifdetectcracker,
  title={SiFDetectCracker: An Adversarial Attack Against Fake Voice Detection Based on Speaker-Irrelative Features},
  author={Hai, Xuan and Liu, Xin and Tan, Yuan and Zhou, Qingguo},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={8552--8560},
  year={2023}
}
```



