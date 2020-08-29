from multiprocessing import set_start_method
set_start_method("spawn")

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
from typing import Tuple, List
import tqdm
import warnings

import multiprocessing as mp
from multiprocessing import Process, Lock, Queue, shared_memory
from multiprocessing.managers import BaseManager, SharedMemoryManager

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers, initializers, optimizers

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))
print("------------------ START ------------------")

eps = np.finfo(np.float32).eps.item()

class ActorCritic(Model):
  def __init__(self, state_size: Tuple, action_size: int):
    super().__init__()

    # Setup Shared Layers for Feature Extraction
    self.shared = keras.Sequential([
      layers.Input(shape=state_size),
      layers.Conv2D(16, 8, strides=4, activation='relu'),
      layers.Conv2D(32, 4, strides=2, activation='relu'),
      layers.Flatten(),
      layers.Dense(256, activation='relu')
    ])

    # Final layers to output actor and critic
    # kernel_init = initializers.RandomNormal(stddev=0.01)
    # self.actor = layers.Dense(action_size, kernel_initializer=kernel_init)
    # self.critic = layers.Dense(1, kernel_initializer=kernel_init)
    self.actor = layers.Dense(action_size)
    self.critic = layers.Dense(1)

  def call(self, inputs):
    print("Inputs", inputs.shape)
    x = self.shared(inputs)
    print(x.shape)
    # x = self.dense(self.conv2(self.conv1(inputs)))
    a, c = self.actor(x), self.critic(x)
    print("AC", a.shape, c.shape)
    return a, c

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
  
  def pull_weights(self):
    self.local_weights = []
    for weight_name, weight_shape in zip(self.weight_names, self.weight_shapes):
      shm = shared_memory.SharedMemory(name=weight_name)
      weight = np.ndarray(weight_shape, dtype=self.weight_dtype, buffer=shm.buf)

      tmp = np.zeros(weight_shape, dtype=self.weight_dtype)
      tmp[:] = weight[:]
      self.local_weights.append(tmp)
    self.local_model.set_weights(self.local_weights)

  def env_step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    state, reward, done, _ = self.env.step(action)
    return (state.astype(np.float32), np.array(reward, np.int32), np.array(done, np.int32))

  def tf_env_step(self, action: tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(self.env_step, [action], [tf.float32, tf.int32, tf.int32])
  
  def run_episode(
    self,
    initial_state: tf.Tensor,  
    model: tf.keras.Model, 
    max_steps: int) -> List[tf.Tensor]:

    action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in tf.range(max_steps):
      # Convert state into a batched tensor (batch size = 1)
      state = tf.expand_dims(state, 0)
    
      # Run the model and to get action probabilities and critic value
      action_logits_t, value = model(state)
    
      # Sample next action from the action probability distribution
      action = tf.random.categorical(action_logits_t, 1)[0, 0]
      action_probs_t = tf.nn.softmax(action_logits_t)

      # Store critic values
      values = values.write(t, tf.squeeze(value))

      # Store log probability of the action chosen
      action_probs = action_probs.write(t, action_probs_t[0, action])
    
      # Apply action to the environment to get next state and reward
      state, reward, done = self.tf_env_step(action)
      state.set_shape(initial_state_shape)
    
      # Store reward
      rewards = rewards.write(t, reward)

      if tf.cast(done, tf.bool):
        break

    action_probs = action_probs.stack()
    values = values.stack()
    rewards = rewards.stack()
    
    return action_probs, values, rewards

  def get_expected_return(
    self,
    rewards: tf.Tensor, 
    gamma: float, 
    standardize: bool = True) -> tf.Tensor:

    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    # Start from the end of `rewards` and accumulate reward sums
    # into the `returns` array
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
      reward = rewards[i]
      discounted_sum = reward + gamma * discounted_sum
      discounted_sum.set_shape(discounted_sum_shape)
      returns = returns.write(i, discounted_sum)
    returns = returns.stack()[::-1]

    if standardize:
      returns = ((returns - tf.math.reduce_mean(returns)) / 
                 (tf.math.reduce_std(returns) + eps))

    return returns

  def compute_loss(
    self,
    action_probs: tf.Tensor,  
    values: tf.Tensor,  
    returns: tf.Tensor) -> tf.Tensor:
    """Computes the combined actor-critic loss."""

    huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

    advantage = returns - values

    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

    critic_loss = huber_loss(values, returns)

    return actor_loss + critic_loss

  @tf.function
  def train_step(
    self,
    initial_state: tf.Tensor, 
    model: tf.keras.Model, 
    optimizer: tf.keras.optimizers.Optimizer, 
    gamma: float, 
    max_steps_per_episode: int) -> tf.Tensor:
    """Runs a model training step."""

    with tf.GradientTape() as tape:
      print("A")
      # Run the model for one episode to collect training data
      action_probs, values, rewards = self.run_episode(
          initial_state, model, max_steps_per_episode) 
      print("A")
      # Calculate expected returns
      returns = self.get_expected_return(rewards, gamma)
      print("A")
      # Convert training data to appropriate TF tensor shapes
      action_probs, values, returns = [tf.expand_dims(x, 1) for x in [action_probs, values, returns]] 
      print("A")

      # Calculating loss values to update our network
      loss = self.compute_loss(action_probs, values, returns)
      print("A")

    # Compute the gradients from the loss
    grads = tape.gradient(loss, model.trainable_variables)
    print("A")

    print(grads)
    print(list(map(lambda x: x.shape, model.trainable_variables)))
    # Apply the gradients to the model's parameters
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print("PLEASE ")

    episode_reward = tf.math.reduce_sum(rewards)

    return episode_reward

  @property
  def exception(self):
    if self._pconn.poll():
      self._exception = self._pconn.recv()
    return self._exception

  def run(self):
    max_episodes = 3
    max_steps_per_episode = 1000

    running_reward = 0
    gamma = 0.99

    with tqdm.trange(max_episodes) as t:
      for i in t:
        print("ITER", i)
        initial_state = tf.constant(self.env.reset(), dtype=tf.float32)
        episode_reward = int(self.train_step(
            initial_state, self.local_model, self.global_opt, gamma, max_steps_per_episode))
        print("ONE STEP")
        running_reward = episode_reward*0.01 + running_reward*.99
      
        t.set_description(f'Episode {i}')
        t.set_postfix(
            episode_reward=episode_reward, running_reward=running_reward)
      
        # Show average episode reward every 10 episodes
        if i % 10 == 0:
          pass # print(f'Episode {i}: average reward: {avg_reward}')

    return

    try:
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
          action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
          probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
          
          # Run simulation until episode done or update time
          done = False
          while not done and thread_step_counter - t_start != args["update_freq"]:
            print("Time", thread_step_counter, t_start)
            # Take action according to policy
            logits, value = self.local_model(tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32))
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
    print("Global")
    print(list(map(lambda x: x.shape, self.global_model.get_weights())))

  def train(self):
    print("----- Train Start -----")

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
      print("----- Worker Start -----")



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

      print("----- Worker Start -----")
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


if __name__ == "__main__":

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



