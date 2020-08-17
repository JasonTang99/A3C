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
    x = self.dense(self.conv2(self.conv1(inputs)))
    return self.actor(x), self.critic(x)

# model = ActorCritic((32, 32, 1), 5)
# model(np.random.random((1, 32, 32, 1)))
# model_weights = model.get_weights()

class Worker(Process):
  global_step_counter = 0
  ma_reward, top_reward = 0, 0, 0
  write_lock = Lock()

  def __init__(self, state_size, action_size, global_model, global_opt, worker_id, write_fp='/tmp'):
    super(Worker, self).__init__()
    self.state_size = state_size
    self.action_size = action_size
    self.local_model = ActorCritic(self.state_size, self.action_size)

    self.global_model = global_model
    self.global_opt = global_opt
    self.worker_id = worker_id
    self.write_fp = write_fp

    self.env = gym.make('SpaceInvaders-v0').unwrapped
    self.episode_loss = 0.0
    self.states = []
    self.actions = []
    self.rewards = []

  def clear_mem(self):
    self.states = []
    self.actions = []
    self.rewards = []

  def run(self):
    thread_step_counter = 1
    
    # Run an episode if the global maximum hasn't stopped
    while Worker.num_episode < args["max_eps"]:
      # Reset and get params?

      t_start = thread_step_counter
      current_state = self.env.reset()

      clear_mem()
      ep_reward = 0.0
      ep_steps = 0
      self.ep_loss = 0.0

      done = False
      while not done:
        logits, _ = self.local_model(tf.convert_to_tensor(current_state[None, :], dtype=tf.float32))
        probs = tf.nn.softmax(logits).numpy()[0]
        action = np.random.choice(len(probs), p=probs)
        
        new_state, reward, done, _ = self.env.step(action)
        if done:
          reward = -1.0
        ep_reward += reward
        self.states.append(current_state)
        self.actions.append(action)
        self.rewards.append(reward)

        if thread_step_counter % args["update_freq"] == 0 or done:
          # Calculate gradient wrt to local model
          with tf.GradientTape() as tape:
            est_reward = 0.0
            if not done:
              est_reward = self.local_model(
                tf.convert_to_tensor(new_state[None, :], dtype=tf.float32)
              )[-1].numpy()[0]

            # Get discounted rewards
            discounted_rewards = []
            for reward in self.rewards[::-1]:  # reverse buffer r
              est_reward = reward + args["gamma"] * est_reward
              discounted_rewards.append(est_reward)
            discounted_rewards[::-1]

            logits, values = self.local_model(
              tf.convert_to_tensor(np.vstack(self.states), dtype=tf.float32)
            )
            # Get our advantages
            advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None], type=tf.float32) - values
            # Value loss
            value_loss = advantage ** 2

            # Calculate our policy loss
            policy = tf.nn.softmax(logits)
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=policy, logits=logits)

            policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.actions, logits=logits)
            policy_loss *= tf.stop_gradient(advantage)
            policy_loss -= 0.01 * entropy
            total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
            return total_loss

            # FIX
            total_loss = self.compute_loss(done, new_state, mem, )


          self.ep_loss += total_loss
          grads = tape.gradient(total_loss, self.local_model.trainable_weights)
          # Apply local gradients to global model
          self.opt.apply_gradients(zip(grads, self.global_model.trainable_weights))
          # Update local model with new weights
          self.local_model.set_weights(self.global_model.get_weights())

          clear_mem()

          if done:
            Worker.global_moving_average_reward = record(
              Worker.global_episode, ep_reward, self.worker_idx,
              Worker.global_moving_average_reward, self.result_queue,
              self.ep_loss, ep_steps
            )

            if global_ep_reward == 0:
              global_ep_reward = episode_reward
            else:
              global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01
            print(f"Episode: {episode} | "
              f"Moving Average Reward: {int(global_ep_reward)} | "
              f"Episode Reward: {int(episode_reward)} | "
              f"Loss: {int(total_loss / float(num_steps) * 1000) / 1000} | "
              f"Steps: {num_steps} | "
              f"Worker: {worker_idx}"
            )
            result_queue.put(global_ep_reward)
            return global_ep_reward

            if ep_reward > Worker.best_score:
              with Worker.save_lock:
                print("Saving best model to {}, "
                      "episode score: {}".format(self.save_dir, ep_reward))
                self.global_model.save_weights(
                    os.path.join(self.save_dir,
                                 'model_{}.h5'.format(self.game_name))
                )
                Worker.best_score = ep_reward
            Worker.global_episode += 1
        ep_steps += 1

        thread_step_counter += 1
        current_state = new_state
    self.result_queue.put(None)

class MasterAgent():
  def __init__(self, args):
    self.game = args["game"]
    self.save_dir = args["save_dir"]
    if not os.path.exists(args["save_dir"]):
      os.makedirs(args["save_dir"])

    env = gym.make(self.game)
    self.state_size = env.observation_space.shape # (210, 160, 3)
    self.action_size = env.action_space.n # 6
    self.opt = tf.compat.v1.train.AdamOptimizer(args["lr"], use_locking=True)

    self.global_model = ActorCritic(self.state_size, self.action_size)
    self.global_model(
        tf.convert_to_tensor(np.random.random(self.state_size), dtype=tf.float32)
    )

  def train(self):
    res_queue = Queue()
    workers = [Worker(
      self.state_size,
      self.action_size,
      self.global_model,
      self.opt, res_queue,
      i, game=self.game,
      save_dir=self.save_dir
    ) for i in range(mp.cpu_count())]

    for i, worker in enumerate(workers):
      print("Starting worker {}".format(i))
      worker.start()

    moving_average_rewards = []
    while True:
      reward = res_queue.get()
      if reward is not None:
        moving_average_rewards.append(reward)
      else:
        break
    [w.join() for w in workers]

    plt.plot(moving_average_rewards)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.savefig(os.path.join(self.save_dir,
                             '{} Moving Average.png'.format(self.game_name)))
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



with SharedMemoryManager() as smm:
  weights_lst = [smm.SharedMemory(size=x.nbytes) for x in model_weights]
  for weight in weights_lst:
    print(weight.name, weight.size)

  p1 = Process(target=do_work, args=(sl, 0, 1000))
  p2 = Process(target=do_work, args=(sl, 1000, 2000))
  p1.start()
  p2.start()  # A multiprocessing.Pool might be more efficient
  p1.join()
  p2.join()   # Wait for all work to complete in both processes
  total_result = sum(sl)  # Consolidate the partial results now in sl

