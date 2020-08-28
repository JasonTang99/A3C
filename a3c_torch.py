import scipy.signal
from random import choice
from time import sleep, time
import gym
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
from typing import Tuple
import warnings
import argparse
import copy

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms


class SharedRMSprop(optim.RMSprop):
    """
    RMSprop with a shared state between processes.
    """
    def __init__(self,
                 params,
                 lr=1e-2,
                 alpha=0.99,
                 weight_decay=0,
                 momentum=0,
                 centered=False):
        super(SharedRMSprop, self).__init__(
            params, lr, alpha, 1e-8, weight_decay, momentum, centered)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['square_avg'] = torch.zeros_like(p.data)
                state['momentum_buffer'] = torch.zeros_like(p.data)
                state['grad_avg'] = torch.zeros_like(p.data)
                
                state['square_avg'].share_memory_()
                state['momentum_buffer'].share_memory_()
                state['grad_avg'].share_memory_()


def train(rank, args, transform, device, global_model, opt, opt_lock, step_counter, ma_reward):
    torch.manual_seed(args.seed + rank)

    # Setup env
    env = gym.make('SpaceInvaders-v0')
    state_size = env.observation_space.shape
    action_size = env.action_space.n

    # Copy Global Model
    local_model = ActorCritic(state_size, action_size).to(device)
    local_model.load_state_dict(global_model.state_dict())

    thread_step_counter = 1
    # Run an episode if the global maximum hasn't been reached
    while step_counter.value < args.max_steps:
        # Sync Models
        local_model.load_state_dict(global_model.state_dict())

        episode_reward, episode_loss = 0, 0
        t_start = thread_step_counter
        state = transform(env.reset())
        
        # Run simulation until episode done or update time
        values, rewards, actions, action_probs = [], [], [], []
        done = False
        while not done and thread_step_counter - t_start != args.update_freq:
            print("Time", t_start, thread_step_counter)
            
            # Take action according to policy
            logits, value = local_model(state.unsqueeze(0))
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).detach()

            next_state, reward, done, _ = env.step(action)
            state = transform(next_state)

            # Store action probs, values, and rewards
            rewards.append(reward)
            values.append(value)
            actions.append(action)
            action_probs.append(probs)
            
            # Update counters
            step_counter.value  += 1
            thread_step_counter += 1

        rewards = torch.tensor(rewards).unsqueeze(1)  # (t, 1)
        values = torch.cat(values, 0)                 # (t, 1)
        actions = torch.cat(actions, 0)               # (t, 1)
        action_probs = torch.cat(action_probs, 0)     # (t, action_space)

        print(rewards)
        print(values)
        print(actions)
        print(action_probs)
        
        # Bootstrap the last state
        R = torch.zeros(1, 1)
        if not done:
            _, value = local_model(state.unsqueeze(0))
            R = value.detach()
        
        # Compute Returns
        returns = []
        for i in range(len(rewards))[::-1]:
            R = rewards[i] + args.gamma * R
            returns.append(R)
        returns = torch.stack(returns[::-1]) # (t, 1)
        
        # Compute values needed in loss calculation
        log_probs = torch.log(action_probs)                     # (t, action_space)
        log_action_probs = torch.gather(log_probs, 1, actions)  # (t, 1)
        advantage = returns - values                            # (t, 1)
        entropy = -torch.sum(action_probs * log_probs)          # (1)
        print(entropy.item(), -torch.sum(log_action_probs * advantage).item())
        # Calculate losses
        actor_loss = -torch.sum(log_action_probs * advantage) - args.beta * entropy
        critic_loss = torch.sum(advantage.pow(2))
        total_loss = actor_loss + critic_loss
        print("Loss", actor_loss.item(), critic_loss.item())

        # Calculate gradient and clip to maximum norm
        total_loss.backward()
        nn.utils.clip_grad_norm_(local_model.parameters(), args.max_grad)
        
        # Propogate gradients to shared model
        with opt_lock:
            for l_param, g_param in zip(local_model.parameters(), global_model.parameters()):
                g_param._grad = l_param.grad.clone()
            opt.step()
            opt.zero_grad()
        local_model.zero_grad()
        # aaa

        if done:
            with torch.no_grad():
                episode_reward = torch.sum(rewards).item()
                episode_loss = total_loss.item()
                if ma_reward.value == 0.0:
                    ma_reward.value = episode_reward
                else:
                    ma_reward.value = ma_reward.value * 0.95 + episode_reward * 0.05

                print("Moving Average Reward: {ma_reward}\t" +\
                      "Episode Reward: {episode_reward}\tLoss: {episode_loss}\t" +
                      "Thread Steps: {thread_step_counter}\tWorker: {worker_id}")

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=8,
            stride=4
        )
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=4,
            stride=2
        )
        self.fc = nn.Linear(13824, 256)

        self.actor = nn.Linear(256, action_size)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # (N, 16, 51, 39)
        x = F.relu(self.conv2(x)) # (N, 32, 24, 18)
        x = F.relu(self.fc(x.view(-1, 13824))) # (N, 256)

        a = self.actor(x) # (N, 6)
        c = self.critic(x) # (N, 1)
        return a, c

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch A3C')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='number of steps to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop alpha (default: 0.99)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='RMSprop weight decay (default: 0)')
    parser.add_argument('--momentum', type=float, default=0.0,
                        help='RMSprop momentum (default: 0)')
    parser.add_argument('--update_freq', type=int, default=5,
                        help='number of steps between syncs (default: 5)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount rate for returns (default: 0.99)')
    parser.add_argument('--beta', type=float, default=0.05,
                        help='scaling factor for entropy (default: 0.05)')
    parser.add_argument('--max_grad', type=float, default=5.0,
                        help='value to clip gradients at (default: 5.0)')
    parser.add_argument('--num-processes', type=int, default=-1,
                        help='number of training processes (default: -1 = cpu count)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    args = parser.parse_args()

    device = torch.device("cpu")
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    torch.manual_seed(args.seed)

    mp.set_start_method('spawn')

    env = gym.make('SpaceInvaders-v0')
    state_size = env.observation_space.shape
    action_size = env.action_space.n

    model = ActorCritic(state_size, action_size).to(device)
    model.share_memory()
    opt = SharedRMSprop(model.parameters(), args.lr, args.alpha, args.weight_decay, args.momentum, False)
    opt_lock = mp.Lock()

    step_counter, ma_reward = mp.Value('d', 0.0), mp.Value('d', 0.0)
    processes = []
    if args.num_processes == -1:
        args.num_processes = mp.cpu_count()
    for rank in range(args.num_processes):
        p = mp.Process(target=train, args=(
            rank, args, transform, device, model, opt, opt_lock, step_counter, ma_reward))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
