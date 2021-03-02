"""
A PyTorch implementation of the A3C (Asynchronous Advantage Actor Critic) paper: 
https://arxiv.org/pdf/1602.01783.pdf
"""
import numpy as np
import gym
import matplotlib.pyplot as plt
import subprocess
import os
import time
import warnings
import argparse
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from shared_optim import SharedRMSprop
from model import ActorCritic
from util import *
from train import *

def run(args):
    device = torch.device("cpu")
    env = gym.make('SpaceInvaders-v0')
    state_size = env.observation_space.shape
    action_size = env.action_space.n

    model = ActorCritic([1, 4, 84, 84], action_size).to(device)
    opt = SharedRMSprop(model.parameters(), lr=args.lr, alpha=args.alpha, 
                        eps=1e-8, weight_decay=args.weight_decay, 
                        momentum=args.momentum, centered=False)
    opt_lock = mp.Lock()
    scheduler = LRScheduler(args)

    if args.load_fp:
        checkpoint = torch.load(args.load_fp)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])

    if args.train:
        start = time.time()

        model.share_memory()
        model.train()

        step_counter, max_reward, ma_reward, ma_loss = [
            mp.Value('d', 0.0) for _ in range(4)]
        
        processes = []
        if args.num_procs == -1:
            args.num_procs = mp.cpu_count()
        for rank in range(args.num_procs):
            p = mp.Process(target=train, args=(
                rank, args, device, model, opt, opt_lock, scheduler,
                step_counter, max_reward, ma_reward, ma_loss))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        if args.verbose > 0:
            print(f"Seconds taken: {time.time() - start}")
        if args.save_fp:
            torch.save({
                'model_state_dict': model.state_dict(),
                # 'optimizer_state_dict': opt.state_dict(),
            }, args.save_fp)
        
    if args.test:
        model.eval()
        test(args, device, model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch A3C')
    parser.add_argument('--steps', type=int, default=1000, metavar='STEPS',
                        help='number of steps to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LEARNING_RATE',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--alpha', type=float, default=0.99, metavar='ALPHA',
                        help='RMSprop alpha (default: 0.99)')
    parser.add_argument('--weight-decay', type=float, default=0.0, metavar='DECAY',
                        help='RMSprop weight decay (default: 0)')
    parser.add_argument('--momentum', type=float, default=0.0, metavar='MOMENTUM',
                        help='RMSprop momentum (default: 0)')
    parser.add_argument('--update-freq', type=int, default=5, metavar='FREQ',
                        help='number of steps between syncs (default: 5)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='GAMMA',
                        help='discount rate for returns (default: 0.99)')
    parser.add_argument('--beta', type=float, default=0.05, metavar='BETA',
                        help='scaling factor for entropy (default: 0.05)')
    parser.add_argument('--max-grad', type=float, default=5.0, metavar='MAX_GRAD',
                        help='value to clip gradients at (default: 5.0)')
    parser.add_argument('--num-procs', type=int, default=-1, metavar='NUM_PROC',
                        help='number of training processes (default: Max)')
    parser.add_argument('--seed', type=int, default=1, metavar='SEED',
                        help='random seed (default: 1)')
    parser.add_argument('--save-fp', type=str, default=None, metavar='SAVE_FP',
                        help='path to save model (default: None)')
    parser.add_argument('--load-fp', type=str, default=None, metavar='LOAD_FP',
                        help='path to load model (default: None)')
    parser.add_argument('--verbose', type=int, default=1, metavar='VERBOSITY',
                        help='verbosity, 2: All, 1: Some, 0: None')
    parser.add_argument('--train', action='store_true', help="perform train")
    parser.add_argument('--test', action='store_true', help="perform test")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    mp.set_start_method('spawn')


    for i in range(20):
        run(args)
        with open(f"run-{i}.txt", "w") as fw:
            with open(f"output", "r") as fr:
                fw.write(fr.read())
        with open(f"output", "w") as fr:
            fr.write("")

        bashCommand = f"cp {args.save_fp} {i}.tar"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        bashCommand = f"cp {os.path.splitext(args.save_fp)[0]}-ckt.tar {i}-ckt.tar"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()