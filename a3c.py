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

import multiprocessing as mp
from multiprocessing import Process, Lock, Queue, shared_memory
from multiprocessing.managers import BaseManager, SharedMemoryManager

import tensorflow as tf
from tensorflow.keras import Model, layers, initializers, optimizers

print("------------------ START ------------------")

class ActorCritic(Model):
  def __init__(self, state_size: Tuple, action_size: int):
    super().__init__()

    # Setup Shared Layers for Feature Extraction
    self.conv1 = layers.Conv2D(16, 8, strides=4, activation='relu', input_shape=state_size)
    self.conv2 = layers.Conv2D(32, 4, strides=2, activation='relu')
    self.dense = layers.Dense(256, activation='relu')

    # Final layers to output actor and critic
    kernel_init = initializers.RandomNormal(stddev=0.01)
    self.actor = layers.Dense(action_size, kernel_initializer=kernel_init)
    self.critic = layers.Dense(1, kernel_initializer=kernel_init)

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    print(inputs.shape)
    print(self.conv1.get_weights())
    try:
      x = self.conv1(inputs)
      print(x.shape)
      x = self.conv2(x)
      print(x.shape)
      x = self.dense(x)
    except:
      print("this isn't good")
    # x = self.dense(self.conv2(self.conv1(inputs)))
    print(x.shape)
    return self.actor(x), self.critic(x)

critic_loss_func = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

class Worker(Process):
  global_step_counter = 0
  global_ma_reward, global_top_reward = 0, 0
  write_lock = Lock()

  def __init__(self, state_size, action_size, weight_names, weight_shapes, weight_dtype, global_opt, results, worker_id, write_fp='/tmp'):
    super(Worker, self).__init__()
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
    self.global_weights = []
    for weight_name, weight_shape in zip(weight_names, weight_shapes):
      memory = shared_memory.SharedMemory(name=weight_name)
      weight = np.ndarray(weight_shape, dtype=weight_dtype, buffer=memory.buf)
      self.global_weights.append(weight)

    self.local_model.set_weights(self.global_weights)
    self.local_model(np.random.random((1, *self.state_size)).astype("float32"))

  def run(self):
    try:
      thread_step_counter = 1
      
      # Run an episode if the global maximum hasn't stopped
      while Worker.global_step_counter < args["max_steps"]:
        # Reset and get params?
        episode_reward, episode_loss = 0, 0
        t_start = thread_step_counter
        state = self.env.reset()
        print("----------- START ENV -----------")
        # Calculate gradient wrt to local model
        with tf.GradientTape() as tape:
          # Create arrays to track and hold our results
          values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
          rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
          action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
          probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
          
          # Run simulation until episode done or update time
          done = False
          while not done and thread_step_counter - t_start != args["update_freq"]:
            print(thread_step_counter, t_start)
            # Take action according to policy
            logits, value = self.local_model(tf.cast(state[np.newaxis, :], tf.float32))
            print(logits, value)
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
            _, value = self.local_model(tf.cast(state[np.newaxis, :], tf.float32))
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
          actor_loss = -tf.math.reduce_sum(action_log_probs * advantage) - args["beta"] * entropy
          
          loss = critic_loss + actor_loss
          episode_loss += loss

        print("----------- APPLY GRAD -----------")
        grads = tape.gradient(total_loss, self.local_model.trainable_weights)
        
        # Apply local gradients to global model
        self.local_model.set_weights(self.global_weights)
        self.global_opt.apply_gradients(zip(grads, self.local_model.trainable_weights))
        
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
          Worker.global_ma_reward =  global_ep_reward

          if episode_reward > Worker.global_top_reward:
            with Worker.write_lock:
              self.global_model.save_weights(self.save_dir + 'model.h5')
              Worker.global_top_reward = episode_reward
    except Exception as e:
      print(e)
      return


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
      ) for i in range(1)]#mp.cpu_count())]

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



