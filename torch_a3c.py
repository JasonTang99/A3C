import scipy.signal
from random import choice
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
from collections import deque
import time

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


def transform(image_in_1, image_in_2):
    # Performs preproocessing of atari environment as described in:
    # Human-level control through deep reinforcement learning by Mnih et al.

    # Take max pixel values between the 2 frames
    image_in = np.maximum(image_in_1, image_in_2)
    # To PIL image
    image_out = Image.fromarray(image_in)
    # RGB -> Luminosity (Grayscale)
    image_out = image_out.convert("L")
    # Resize
    image_out = image_out.resize((84, 84))
    # Back to Tensor
    return transforms.functional.to_tensor(image_out)

def train(rank, args, device, global_model, opt, opt_lock, 
          step_counter, ma_reward, ma_loss):
    torch.manual_seed(args.seed + rank)

    # Setup env
    env = gym.make('SpaceInvaders-v0')
    state_size = env.observation_space.shape
    action_size = env.action_space.n

    state = env.reset()
    next_state, _, _, _ = env.step(0)

    # Keep a deque of 4 processed states as input
    proc_state = transform(state, next_state)
    proc_states = deque([
        proc_state.clone(), proc_state.clone(),
        proc_state.clone(), proc_state.clone()
    ], maxlen=4)
    state = next_state

    # Setup local model
    local_model = ActorCritic(state_size, action_size).to(device)

    thread_step_counter = 1
    ep_reward, ep_loss = 0.0, 0.0
    # Run an episode if the global maximum hasn't been reached
    while step_counter.value < args.max_steps:
        # Sync Models
        local_model.load_state_dict(global_model.state_dict())

        # Run simulation until episode done or update time
        values, rewards, actions, action_probs = [], [], [], []
        t_start = thread_step_counter
        done = False

        while not done and thread_step_counter - t_start != args.update_freq:
            # Take action according to policy
            processed_input = torch.cat(tuple(proc_states)).unsqueeze(0)
            logits, value = local_model(processed_input)
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).detach()
            next_state, reward, done, _ = env.step(action)

            # Update states
            proc_state = transform(state, next_state)
            proc_states.append(proc_state)
            state = next_state

            # Store action probs, values, and rewards
            rewards.append(reward)
            values.append(value)
            actions.append(action)
            action_probs.append(probs)

            # Update counters
            step_counter.value += 1
            thread_step_counter += 1

        # if step_counter.value % 100 == 0:
        #     print("{:.2f}%".format(100 * step_counter.value / args.max_steps))
        # env.render()

        rewards = torch.tensor(rewards).unsqueeze(1)  # (t, 1)
        values = torch.cat(values, 0)                 # (t, 1)
        actions = torch.cat(actions, 0)               # (t, 1)
        action_probs = torch.cat(action_probs, 0)     # (t, action_space)

        # Bootstrap the last state
        R = torch.zeros(1, 1)
        if not done:
            processed_input = torch.cat(tuple(proc_states)).unsqueeze(0)
            _, value = local_model(processed_input)
            R = value.detach()

        # Compute Returns
        returns = []
        for i in range(len(rewards))[::-1]:
            R = rewards[i] + args.gamma * R
            returns.extend(R)
        returns = torch.stack(returns[::-1])  # (t, 1)

        # Compute values needed in loss calculation
        t = action_probs.shape[0]
        log_probs = torch.log(action_probs)                     # (t, action_space)
        log_action_probs = torch.gather(log_probs, 1, actions)  # (t, 1)
        advantage = returns - values                            # (t, 1)
        entropy = -torch.sum(action_probs * log_probs) / t      # (1)

        # Calculate losses
        actor_loss = -torch.sum(log_action_probs *
                                advantage) - args.beta * entropy
        critic_loss = torch.sum(advantage.pow(2))
        total_loss = actor_loss + critic_loss

        # Calculate gradient and clip to maximum norm
        total_loss.backward()
        nn.utils.clip_grad_norm_(local_model.parameters(), args.max_grad)

        # Propogate gradients to shared model
        with opt_lock:
            for l_param, g_param in zip(local_model.parameters(), 
                                        global_model.parameters()):
                g_param._grad = l_param.grad.clone()
            opt.step()
            opt.zero_grad()
        local_model.zero_grad()

        with torch.no_grad():
            ep_reward += torch.sum(rewards).item()
            ep_loss += total_loss.item()
        if done:
            with torch.no_grad():
                if ma_reward.value == 0.0:
                    ma_reward.value = ep_reward
                    ma_loss.value = ep_loss
                else:
                    ma_reward.value = ma_reward.value * 0.95 + ep_reward * 0.05
                    ma_loss.value = ma_loss.value * 0.95 + ep_loss * 0.05

                print(f"MA Reward: {ma_reward.value:.2f}\t" +
                      f"MA Loss: {ma_loss.value:.2f}\t" +
                      f"EP Reward: {ep_reward}  \tEP Loss: {ep_loss:.4E}\t" +
                      f"Thread-{rank} Steps: {thread_step_counter}")

                ep_reward, ep_loss = 0.0, 0.0

                # Reset Environment
                state = env.reset()
                next_state, _, _, _ = env.step(0)

                # Keep a deque of 4 processed states as input
                proc_state = transform(state, next_state)
                proc_states = deque([
                    proc_state.clone(), proc_state.clone(),
                    proc_state.clone(), proc_state.clone()
                ], maxlen=4)
                state = next_state


def test(args, device, model, tries=3, max_steps=1000000):
    torch.manual_seed(args.seed)

    # Setup env
    env = gym.make('SpaceInvaders-v0')
    state_size = env.observation_space.shape
    action_size = env.action_space.n

    # Run the simulation "tries" times
    for t in range(tries):
        # Reset Env
        state = env.reset()
        next_state, _, _, _ = env.step(0)

        # Keep a deque of 4 processed states as input
        proc_state = transform(state, next_state)
        proc_states = deque([
            proc_state.clone(), proc_state.clone(),
            proc_state.clone(), proc_state.clone()
        ], maxlen=4)
        state = next_state

        # Run simulation until episode done
        images = []
        ep_reward = 0.0
        for i in range(max_steps):
            if i % 3 == 0:
                env.render()
                screen = env.render(mode='rgb_array')
                images.append(Image.fromarray(screen))
            
            # Take action according to policy
            processed_input = torch.cat(tuple(proc_states)).unsqueeze(0)
            logits, value = model(processed_input)
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).detach()
            next_state, reward, done, _ = env.step(action)

            # Update states
            proc_state = transform(state, next_state)
            proc_states.append(proc_state)
            state = next_state

            ep_reward += reward
            if done:
                break
        
        # Save as gif
        print(f'Try #{t} reward: {ep_reward}')
        images[0].save(f'invaders-{t}.gif', save_all=True,
                    append_images=images[1:], loop=0, duration=1)
    

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=4,
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
        self.fc = nn.Linear(2592, 256)

        self.actor = nn.Linear(256, action_size)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (N, 16, 20, 20)
        x = F.relu(self.conv2(x))  # (N, 32, 9, 9)
        x = F.relu(self.fc(x.view(x.shape[0], -1)))  # (N, 256)

        a = self.actor(x)  # (N, 6)
        c = self.critic(x)  # (N, 1)
        return a, c


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch A3C')
    parser.add_argument('--max_steps', type=int, default=10000,
                        help='number of steps to train (default: 10000)')
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
                        help='number of training processes (default: Max)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--save-fp', type=str, default=None,
                        help='path to save model (default: None)')
    parser.add_argument('--load-fp', type=str, default=None,
                        help='path to load model (default: None)')
    parser.add_argument('--method', type=str, default="train",
                        help='train or test (default: train)')
    args = parser.parse_args()

    device = torch.device("cpu")
    torch.manual_seed(args.seed)
    mp.set_start_method('spawn')

    env = gym.make('SpaceInvaders-v0')
    state_size = env.observation_space.shape
    action_size = env.action_space.n

    model = ActorCritic(state_size, action_size).to(device)
    if args.load_fp:
        model.load_state_dict(torch.load(args.load_fp))

    if args.method == "train":
        start = time.time()

        model.share_memory()
        model.train()

        opt = SharedRMSprop(model.parameters(), args.lr, args.alpha,
                            args.weight_decay, args.momentum, False)
        opt_lock = mp.Lock()

        step_counter, ma_reward, ma_loss = mp.Value(
            'd', 0.0), mp.Value('d', 0.0), mp.Value('d', 0.0)
        processes = []
        if args.num_processes == -1:
            args.num_processes = mp.cpu_count()
        for rank in range(args.num_processes):
            p = mp.Process(target=train, args=(
                rank, args, device, model, opt, opt_lock, 
                step_counter, ma_reward, ma_loss))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        print(f"Seconds taken: {time.time() - start}")
        if args.save_fp:
            torch.save(model.state_dict(), args.save_fp)
    else:
        test(args, device, model)

