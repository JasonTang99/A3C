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
                 eps=1e-8,
                 weight_decay=0,
                 momentum=0,
                 centered=False):
        super(SharedRMSprop, self).__init__(
            params, lr, alpha, eps, weight_decay, momentum, centered)

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

def train(rank, args, global_model, global_opt, device, write_fp):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    local_model = copy.deepcopy(global_model)
    

    for epoch in range(1, args.epochs + 1):        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data.to(device))
            loss = F.nll_loss(output, target.to(device))
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    rank, epoch,
                    batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()
                ))
                if args.dry_run:
                    break

    global_step_counter, global_ma_reward, global_top_reward = 0, 0, 0
    write_lock = Lock()

    self.state_size = state_size
    self.action_size = action_size
    self.local_model = ActorCritic(self.state_size, self.action_size)
    self.local_model(np.random.random((1, *self.state_size)).astype("float32"))

    self.global_opt = global_opt
    self.results = results
    self.worker_id = worker_id
    self.write_fp = write_fp

    self.env = gym.make('SpaceInvaders-v0').unwrapped
    self.episode_loss = 0.0

    self.weight_names = weight_names
    self.weight_shapes = weight_shapes
    self.weight_dtype = weight_dtype

    print("WORKER LOAD START")
    print(list(map(lambda x: x.shape, self.local_model.get_weights())))

    self.local_weights = []
    self.pull_weights()

    print("WORKER LOAD MIDDLE")
    print(list(map(lambda x: x.shape, self.local_model.get_weights())))

    self.local_model(np.random.random((1, *self.state_size)).astype("float32"))

    print("WORKER LOAD")

    def run(self):
        thread_step_counter = 1

        # Run an episode if the global maximum hasn't stopped
        while Worker.global_step_counter < args["max_steps"]:
            # Reset and get params?
            episode_reward, episode_loss = 0, 0
            t_start = thread_step_counter
            state = self.env.reset()
            print("----------- START ENV -----------")
            print(state.shape)
            # Calculate gradient wrt to local model
            with tf.GradientTape() as tape:
            # Create arrays to track and hold our results
                values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
                rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
                action_probs = tf.TensorArray(
                    dtype=tf.float32, size=0, dynamic_size=True)
                probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

                # Run simulation until episode done or update time
                done = False
                while not done and thread_step_counter - t_start != args["update_freq"]:
                    print("Time", thread_step_counter, t_start)
                    # Take action according to policy
                    logits, value = self.local_model(tf.convert_to_tensor(
                        state[np.newaxis, :], dtype=tf.float32))
                    print(logits[0], value[0])
                    probs = tf.nn.softmax(logits).numpy()[0]
                    action = np.random.choice(len(probs), p=probs)
                    next_state, reward, done, _ = self.env.step(action)

                    print(reward)

                    # Store action probs, values, and rewards
                    t = thread_step_counter - t_stasrt
                    values = values.write(t, value[0])
                    rewards = rewards.write(t, reward)
                    action_probs = action_probs.write(t, probs[action])
                    probs = probs.write(t, probs)

                    episode_reward += reward
                    thread_step_counter += 1
                    state = next_state

                values = values.stack()
                rewards = rewards.stack()
                action_probs = action_probs.stack()
                probs = probs.stack()

                print("----------- START GRAD -----------")

                # Final reward is 0 if done, else bootstrap with V(s)
                discounted_sum = tf.constant(0.0)
                if not done:
                    _, value = self.local_model(
                        tf.cast(state[np.newaxis, :], tf.float32))
                    discounted_sum += value.numpy()[0]

                # Compute Returns
                returns = tf.TensorArray(dtype=tf.float32, size=tf.shape(rewards)[0])
                rewards = rewards[::-1]
                for i in tf.range(tf.shape(rewards)[0]):
                    discounted_sum = rewards[i] + gamma * discounted_sum
                    returns = returns.write(i, discounted_sum)
                returns = returns.stack()[::-1]

                if False:
                    returns = ((returns - tf.math.reduce_mean(returns)) /
                            (tf.math.reduce_std(returns) + eps))

                # Advantage
                advantages = returns - values

                # Critic loss
                critic_loss = critic_loss_func(values, returns)

                # Actor loss
                action_log_probs = tf.math.log(action_probs)
                entropy = -tf.math.reduce_sum(probs * tf.math.log(probs))
                actor_loss = - \
                    tf.math.reduce_sum(action_log_probs *
                                        advantage) - args["beta"] * entropy

                loss = critic_loss + actor_loss
                episode_loss += loss

            print("----------- APPLY GRAD -----------")
            grads = tape.gradient(total_loss, self.local_model.trainable_weights)

            # Apply local gradients to global model
            self.local_model.set_weights(self.global_weights)
            self.global_opt.apply_gradients(
                zip(grads, self.local_model.trainable_weights))

            # Update global model with new weights
            for local_weight, global_weight in (self.local_model.get_weights(), self.global_weights):
                global_weight[:] = local_weight[:]

            if done:
                if Worker.global_ma_reward == 0:
                    Worker.global_ma_reward = episode_reward
                else:
                    Worker.global_ma_reward = Worker.global_ma_reward * 0.95 + episode_reward * 0.05

                print("Moving Average Reward: {}\tEpisode Reward: {}\tLoss: {}\tThread Steps: {}\tWorker:{}".format(
                    Worker.global_ma_reward, episode_reward, episode_loss, thread_step_counter, worker_id))

                self.results.put(Worker.global_ma_reward)
                Worker.global_ma_reward = global_ep_reward

                if episode_reward > Worker.global_top_reward:
                    with Worker.write_lock:
                        self.global_model.save_weights(self.save_dir + 'model.h5')
                        Worker.global_top_reward = episode_reward

def test(args, model, device, dataset, dataloader_kwargs):
    test_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data.to(device))
            test_loss += F.nll_loss(output, target.to(device), reduction='sum').item() # sum up batch loss
            pred = output.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.to(device)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


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
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                        help='how many training processes to use (default: 2)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    args = parser.parse_args()

    device = torch.device("cpu")
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    torch.manual_seed(args.seed)

    env = gym.make('SpaceInvaders-v0')
    state_size = env.observation_space.shape
    action_size = env.action_space.n

    mp.set_start_method('spawn')
    model = ActorCritic(state_size, action_size).to(device)
    opt = SharedRMSprop(model.parameters())
    print("MADE OPT")
    test_in = np.random.randint(0, 256, state_size, dtype=np.uint8)
    test_in = transform(test_in)
    test_in = test_in.view(-1, *test_in.shape)
    print(model(test_in))

    model.share_memory() # gradients are allocated lazily, so they are not shared here

    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=train, args=(rank, args, model, device, dataset1, kwargs))
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # Once training is complete, we can test the model
    test(args, model, device, dataset2, kwargs)


class Master():
  def __init__(self):
    self.save_dir = args["save_dir"]
    if not os.path.exists(args["save_dir"]):
      os.makedirs(args["save_dir"])

    env = gym.make('SpaceInvaders-v0')
    self.state_size = env.observation_space.shape # (210, 160, 3)
    self.action_size = env.action_space.n # 6
    self.opt = tf.compat.v1.train.AdamOptimizer(args["lr"], use_locking=True)

    # Initialize the global model
    self.global_model = ActorCritic(self.state_size, self.action_size)
    self.global_model(np.random.random((1, *self.state_size)).astype("float32"))
    print("Global")
    print(list(map(lambda x: x.shape, self.global_model.get_weights())))

  def train(self):
    with SharedMemoryManager() as smm:
      res_queue = Queue()

      model_weights = self.global_model.get_weights()
      weight_memory = [smm.SharedMemory(size=x.nbytes) for x in model_weights]
      weight_names, weight_shapes, weight_dtype = [], [], None
      shared_weights = []
      for memory, weight in zip(weight_memory, model_weights):
        print(memory.name)
        new_weight = np.ndarray(weight.shape, dtype=weight.dtype, buffer=memory.buf)
        new_weight[:] = weight[:]
        weight_names.append(memory.name)
        weight_shapes.append(weight.shape)
        weight_dtype = weight.dtype
        shared_weights.append(new_weight)

      workers = [Worker(
        self.state_size,
        self.action_size,
        weight_names,
        weight_shapes, 
        weight_dtype,
        self.opt, 
        res_queue,
        i, 
        write_fp=self.save_dir
      ) for i in range(1)] #mp.cpu_count())]

      for worker in workers:
        worker.start()
      for worker in workers:
        worker.join()

    res_queue_items = []
    try:
      while True:
        res_queue_items.append(res_queue.get())
    except:
      pass

    print(res_queue_items)
    plt.plot(res_queue_items)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.savefig(self.save_dir + 'Moving Average.png')
    plt.show()

  def play(self):
    env = gym.make(self.game_name).unwrapped
    state = env.reset()
    model = self.global_model
    model_path = os.path.join(self.save_dir, 'model_{}.h5'.format(self.game_name))
    print('Loading model from: {}'.format(model_path))
    model.load_weights(model_path)
    done = False
    step_counter = 0
    reward_sum = 0

    try:
      while not done:
        env.render(mode='rgb_array')
        policy, value = model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
        policy = tf.nn.softmax(policy)
        action = np.argmax(policy)
        state, reward, done, _ = env.step(action)
        reward_sum += reward
        print("{}. Reward: {}, action: {}".format(step_counter, reward_sum, action))
        step_counter += 1
    except KeyboardInterrupt:
      print("Received Keyboard Interrupt. Shutting down.")
    finally:
      env.close()

args = {
  "lr": 0.0005,
  "save_dir": ".",
  "gamma": 0.99,
  "beta": 0.1,
  "max_steps": 100,
  "update_freq": 5
}
master = Master()
master.train()



