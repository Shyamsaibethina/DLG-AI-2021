import numpy as np
import pickle
import gym
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.initializers as initializers
import random
#hello
# learning rate to try: 0.00025
class Memory:
    def __init__(self, max_memory):
        self.max_memory = max_memory
        self.samples = []
    
    def add_sample(self, sample):
        self.samples.append(sample)
        if len(self.samples) > self.max_memory:
            self.samples.pop(0)
            
    def sample(self, no_samples):
        if no_samples > len(self.samples):
            return random.sample(self.samples, len(self.samples))
        else:
            return random.sample(self.samples, no_samples)

initializer = initializers.GlorotNormal

x_in = layers.Input(shape = (6400,))
x = layers.Dense(200, kernel_initializer= initializer, activation="relu")(x_in)
x_out = layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(x_in, x_out) # Use this for fitting as epsilon will bring it all over the damm place

model.compile(optimizer = "adam", loss = "mse", metrics = ["acc"])

target_model = model # Use this one for prediction to have some semblence of consistency

target_model.set_weights(model.get_weights())

model.summary()

class GameRunner:
    def __init__(self, env, model, target_model, memory, epsilon, max_eps, min_eps, game_dimensions, epsilon_greedy_frames, resume = False, render = True):
        self.env = env
        self.model = model
        self.target_model = target_model
        self.memory = memory
        self.eps = epsilon
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.render = render
        self.resume = resume
        self.epsilon_greedy_frames = epsilon_greedy_frames
        self.gameDimensions = game_dimensions
        self.rewards = []
        self.max_x = []
    
    def run(self):
        observation = self.env.reset()
        reward_sum = 0
        running_reward = None
        prev_frame = None
        episode_number = 0
        
        while True:
            if self.render:
                env.render()

            if self.resume:
                self.model.load_weights("ModelWeights")
            
            curr_frame = self.prepro(observation)
            # change_in_frame = curr_frame - prev_frame if prev_frame is not None else np.zeros(self.gameDimensions)
            
            action = self.choose_action(curr_frame)
            
            observation, reward, done, _ = self.env.step(3) 
            
            next_frame = self.prepro(observation)
            
            y = 1 if action == 2 else 0
            
            curr_frame = next_frame
            
            self.memory.add_sample((curr_frame, y, reward, next_frame, done))
            self.replay(done)
            
            if episode_number % 10 == 0: # Probably should adjust this number
                self.target_model.set_weights(self.model.get_weights())
                self.target_model.save_weights("ModelWeights")
            
            # Decay probability of taking random action
            epsilon_interval = (self.max_eps - self.min_eps)
            self.eps -= epsilon_interval / self.epsilon_greedy_frames
            self.eps = max(self.eps, self.min_eps)
            
            reward_sum += reward
            
            if done:
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print ('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
                reward_sum = 0
                observation = env.reset() # reset env
                prev_frame = None
                episode_number += 1

            if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
                print('ep %d: game finished, reward: %f, epsilon: %g' % (episode_number, reward, self.eps) + ('' if reward == -1 else ' !!!!!!!!'))
                
    def choose_action(self, state):
        if np.random.random() < self.eps:
                return np.random.randint(2, 4)
        else:
            state = state.reshape((1,6400))
            up_prob = self.target_model.predict(state)
            if up_prob >= .5:
                return 2
            else:
                return 3
            
    def prepro(self, input_frame):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        input_frame = input_frame[34:194] # crop
        input_frame = input_frame[::2,::2,0] # downsample by factor of 2 (halves the resolution of the image)
        #This takes every other pixel in the image
        input_frame[input_frame == 144] = 0 # erase background (background type 1)
        input_frame[input_frame == 109] = 0 # erase background (background type 2)
        input_frame[input_frame != 0] = 1 # everything else (paddles, ball) just set to 1
        return input_frame.astype(np.float).ravel()
    
    def replay(self, terminal_state):
        gamma = 0.99
        batches = self.memory.sample(500)
        states = np.array([val[0] for val in batches])
        rewards = np.array([val[2] for val in batches])
        new_current_states = np.array([val[3] for val in batches])
        
        q_s_a = self.model.predict(states)
        
        future_q_s_a = self.target_model.predict(new_current_states)
        
        x = []
        y = []
        
        for i, b in enumerate(batches):
            current_state, action, reward, new_state = b[0], b[1], b[2], b[3]
            
            if not b[4]:
                max_future_q = np.max(future_q_s_a[i])
                new_q = reward + gamma * max_future_q
            else:
                new_q = reward
            
            current_qs = q_s_a[0]
            current_qs[0] = new_q
            
            x.append(current_state)
            y.append(current_qs)
        
        self.model.fit(np.asarray(x),np.asarray(y), batch_size = len(batches), verbose = 1 if terminal_state else None)

batch_size = 10

env = gym.make("Pong-v0")

mem = Memory(50_000)

eps = 1.0
max_eps = 1.0
min_eps = 0.000001
eps_greedy_frames = 100000.0

game_dimensions = 80*80

gr = GameRunner(env, model, target_model, mem, eps, max_eps, min_eps, game_dimensions, eps_greedy_frames, resume = False, render = False)

gr.run()
env.observation_space
env.unwrapped.get_action_meanings()

# NOOP is the same as FIRE (standing still)
# LEFT is the same as LEFTFIRE (down)
# RIGHT is the same as RIGHTFIRE (up)

env.unwrapped.get_keys_to_action()