import argparse
import numpy as np
import os
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from models.model import Network
from func.functions import evaluate
from utils.utils import Genotype
from ASVRawDataset import ASVRawDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ASVSpoof2019 model')
    parser.add_argument('--data', type=str, default='/home/dslab/hx/vfuzz/media/data/LA', 
                    help='location of the data corpus')   
    parser.add_argument('--model', default='/home/dslab/hx/vfuzz/Nattack/test_model/raw-pc-darts-anti-spoofing/epoch_66.pth', type=str)
    parser.add_argument('--layers', type=int, default=8)
    parser.add_argument('--init_channels', type=int, default=64)
    parser.add_argument('--gru_hsize', type=int, default=1024)
    parser.add_argument('--gru_layers', type=int, default=3)
    parser.add_argument('--sinc_scale', type=str, default='mel', help='the ytpe of sinc layer')
    parser.add_argument('--sinc_kernel', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--sr', type=int, default=16000, help='default sampling rate')
    parser.add_argument('--arch', type=str, default="Genotype(normal=[('dil_conv_5', 1), ('dil_conv_3', 0), ('dil_conv_5', 1), ('dil_conv_5', 2), ('std_conv_5', 2), ('skip_connect', 3), ('std_conv_5', 2), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3', 0), ('std_conv_3', 1), ('dil_conv_3', 0), ('dil_conv_3', 2), ('skip_connect', 0), ('dil_conv_5', 2), ('dil_conv_3', 0), ('avg_pool_3', 1)], reduce_concat=range(2, 6))", help='the searched architecture')
    parser.add_argument('--comment', type=str)
    parser.add_argument('--eval', type=str, default='e', help='to use eval or dev')

    parser.set_defaults(is_mask=False)
    parser.set_defaults(is_trainable=False)
    
    args = parser.parse_args()
    OUTPUT_CLASSES = 2
    checkpoint = torch.load(args.model)
    genotype = eval(args.arch)


    model = Network(args.init_channels, args.layers, args, OUTPUT_CLASSES, genotype)
    model.drop_path_prob = 0.0

    if args.eval == 'e':
        eval_protocol = 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
        eval_dataset = ASVRawDataset(Path(args.data), 'eval', eval_protocol, is_rand=False)
    elif args.eval == 'd':
        print('*'*50)
        print('using dev protocol...')
        eval_protocol = 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
        eval_dataset = ASVRawDataset(Path(args.data), 'dev', eval_protocol, is_rand=False)
        print('*'*50)


    model = model.cuda()
    model.load_state_dict(checkpoint)

    eval_loader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    evaluate(eval_loader, model, args.comment)
    print('Done')
