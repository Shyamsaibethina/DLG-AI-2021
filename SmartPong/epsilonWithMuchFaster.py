import numpy as np
import pickle
import gym
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.initializers as initializers
#from support import save_frames_as_gif
from matplotlib import animation


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
'''def save_frames_as_gif(frames, filename=None):
    """
    Save a list of frames as a gif
    """ 
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    if filename:
        anim.save(filename, dpi=72, writer='imagemagick')
'''

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
epsilon_greedy_frames = 100000.0 #lowering this value makes epsilon decrease faster
#!!!!!!!frames = []
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
    #frames.append(observation)
    
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
    #print("epsilon is: ", epsilon)
    if done:
        comb_ep_observations = np.vstack(ep_observations)
        comb_ep_gradient_log_ps = np.vstack(ep_gradient_log_ps)
        comb_ep_rewards = np.vstack(ep_rewards)
        ep_observations, ep_gradient_log_ps, ep_rewards = [], [], []
        
        discounted_comb_ep_rewards = discount_rewards(comb_ep_rewards)
        discounted_comb_ep_rewards -= np.mean(discounted_comb_ep_rewards) 
        discounted_comb_ep_rewards /= np.std(discounted_comb_ep_rewards)
        
        comb_ep_gradient_log_ps = (comb_ep_gradient_log_ps * discounted_comb_ep_rewards)
        
        model.fit(comb_ep_observations, comb_ep_gradient_log_ps, epochs = 50)
        
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
    #save_frames_as_gif(frames, filename = 'gif1')