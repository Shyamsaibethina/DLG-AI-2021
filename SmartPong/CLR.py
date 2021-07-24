#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pickle
import gym
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.initializers as initializers
#hello
env = gym.make("Pong-v0")
initializer = initializers.GlorotNormal

x_in = layers.Input(shape = (6400,))
x = layers.Dense(200, kernel_initializer= initializer, activation="relu")(x_in)
x_out = layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(x_in, x_out)

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["acc"])
env.observation_space
env.unwrapped.get_action_meanings()

# NOOP is the same as FIRE (standing still)
# LEFT is the same as LEFTFIRE (down)
# RIGHT is the same as RIGHTFIRE (up)
model.summary()
def prepro(input_frame):
    """ prepro 210x160x3 uint8 frame into e6400 (80x80) 1D float vector """
    input_frame = input_frame[34:194] # crop
    input_frame = input_frame[::2,::2,0] # downsample by factor of 2 (halves the resolution of the image)
    #This takes every other pixel in the image
    input_frame[input_frame == 144] = 0 # erase background (background type 1)
    input_frame[input_frame == 109] = 0 # erase background (background type 2)
    input_frame[input_frame != 0] = 1 # everything else (paddles, ball) just set to 1
    return input_frame.astype(np.float).ravel()
    
def discount_rewards(rewards):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, rewards.size)):
        if rewards[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add
    return discounted_r

render = False
prev_frame = None
game_dimensions = 80*80
gamma = 0.99
resume = False
batch_size = 10

observation = env.reset()
reward_sum = 0
epsilon = 1  # Epsilon greedy parameter
epsilon_min = 0.000001  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)
episode_number = 0
running_reward = None
ep_observations, ep_rewards, ep_gradient_log_ps = [], [], []
# Number of episodes to take random action and observe output
epsilon_random_episodes = 50000  #we can set a maximum number of episodes for exploration using this variable
# Number of frames for exploration
epsilon_greedy_frames = 50000.0 #lowering this value makes epsilon decrease faster

if resume:
    model.load_weights("ModelWeights")

while True:
    if render:
        env.render()
    
    curr_frame = prepro(observation)
    change_in_frame = curr_frame - prev_frame if prev_frame is not None else np.zeros(game_dimensions)
    prev_frame = curr_frame
    ep_observations.append(change_in_frame)
    
    if episode_number < epsilon_random_episodes or np.random.random() < epsilon:
        action = np.random.randint(2, 4)
    else:
        change_in_frame = change_in_frame.reshape((1,6400))
        up_prob = model.predict(change_in_frame)
        if up_prob >= .5:
            action = 2
        else:
            action = 3
    
    observation, reward, done, _ = env.step(action) 
    
    y = 1 if action == 2 else 0
    
    try:
        ep_gradient_log_ps.append(y - up_prob)
    except:
        ep_gradient_log_ps.append(y)
    
    ep_rewards.append(reward)
    reward_sum += reward
    # Decay probability of taking random action
    epsilon -= epsilon_interval / epsilon_greedy_frames
    epsilon = max(epsilon, epsilon_min) #this just makes sure you never go below the minimum desired epsilon
#     print("epsilon is: ", epsilon)
    
    if done:
        comb_ep_observations = np.vstack(ep_observations)
        comb_ep_gradient_log_ps = np.vstack(ep_gradient_log_ps)
        comb_ep_rewards = np.vstack(ep_rewards)
        ep_observations, ep_gradient_log_ps, ep_rewards = [], [], []
        
        discounted_comb_ep_rewards = discount_rewards(comb_ep_rewards)
        discounted_comb_ep_rewards -= np.mean(discounted_comb_ep_rewards) 
        discounted_comb_ep_rewards /= np.std(discounted_comb_ep_rewards)
        
        comb_ep_gradient_log_ps = (comb_ep_gradient_log_ps * discounted_comb_ep_rewards)
        
        """" CLR """
        
        clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
        
        model.fit(comb_ep_observations, comb_ep_gradient_log_ps, epochs = 50, callbacks = [clr])
        
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print ('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        reward_sum = 0
        observation = env.reset() # reset env
        prev_frame = None
        episode_number += 1
        
        if episode_number % batch_size == 0:
            
            # Back Propagation Code Goes Here
            
            model.save_weights("ModelWeights")
    
    if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
        print('ep %d: game finished, reward: %f' % (episode_number, reward) + ('' if reward == -1 else ' !!!!!!!!'))


# In[2]:


from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
import numpy as np

class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())


# In[ ]:





# In[ ]:




