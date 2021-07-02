""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle #needed to be able to visualize results when you win. 
import gym
import time


# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-3 # a lot of the comments were saying to change this to make it faster
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = True

# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid
if resume:
  model = pickle.load(open('save.p', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = np.random.randn(H) / np.sqrt(H) #Divide by square root of number of dimension size to normalize weights

grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory
# np.zeros_like - Return an array of zeros with the same shape and type as a given array.

def sigmoid(x): 
    return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]


def prepro(input_frame):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  input_frame = input_frame[35:195] # crop
  input_frame = input_frame[::2,::2,0] # downsample by factor of 2 (halves the resolution of the image)
  input_frame[input_frame == 144] = 0 # erase background (background type 1)
  input_frame[input_frame == 109] = 0 # erase background (background type 2)
  input_frame[input_frame != 0] = 1 # everything else (paddles, ball) just set to 1
  return input_frame.astype(np.float).ravel()

def discount_rewards(rewards):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(rewards)
  running_add = 0
  # print(rewards)  r is a 2d array with the given rewards
  for t in reversed(range(0, rewards.size)):
    if rewards[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + rewards[t]
    discounted_r[t] = running_add
  return discounted_r

# TODO: understand and rename (if needed) logp
def policy_forward(change_in_frame):
  hidden_layer_values = np.dot(model['W1'], change_in_frame)
  hidden_layer_values[hidden_layer_values<0] = 0 # ReLU nonlinearity
  logp = np.dot(model['W2'], hidden_layer_values)
  up_prob = sigmoid(logp)
  return up_prob, hidden_layer_values # return probability of taking action 2, and hidden state

# TODO: understand and rename (if needed) dh
# dC_dw1: derivative of the cost / derivative of the weights 1 --> partial derivative
# dC_dw2: derivative of the cost / derivative of the weights 2 --> partial derivative
#comb_ep_hidden_layer_values.T and dh.T are the transposed matrices of comb_ep_hidden_layer_values and dh, respectively
def policy_backward(comb_ep_hidden_layer_values, comb_ep_gradient_log_ps):
  """ backward pass. (comb_ep_hidden_layer_values is array of intermediate hidden states) """
  dC_dw2 = np.dot(comb_ep_hidden_layer_values.T, comb_ep_gradient_log_ps).ravel()
  dh = np.outer(comb_ep_gradient_log_ps, model['W2'])
  dh[comb_ep_hidden_layer_values <= 0] = 0 # backpro prelu
  dC_dw1 = np.dot(dh.T, comb_ep_observations)
  return {'W1':dC_dw1, 'W2':dC_dw2}

env = gym.make("Pong-v0")
observation = env.reset() # gets very first image of the game
prev_frame = None # used in computing the difference frame 
ep_observations,ep_hidden_layer_values,ep_gradient_log_ps,ep_rewards = [],[],[],[]

running_reward = None
reward_sum = 0
episode_number = 0

while True:
  if render: 
    env.render()
    #time.sleep(0.5)

  # preprocess the observation, set input to network to be difference image
  curr_frame = prepro(observation)
  change_in_frame = curr_frame - prev_frame if prev_frame is not None else np.zeros(D)
  prev_frame = curr_frame

  # forward the policy network and sample an action from the returned probability
  up_prob, hidden_layer_values = policy_forward(change_in_frame)
  action = 2 if np.random.uniform() < up_prob else 3 # roll the dice!

  # record various intermediates (needed later for backprop)
  ep_observations.append(change_in_frame) # observation
  ep_hidden_layer_values.append(hidden_layer_values) # hidden state
  y = 1 if action == 2 else 0 # a "fake label"
  ep_gradient_log_ps.append(y - up_prob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

  # step the environment and get new measurements
  observation, reward, done, info = env.step(action)
  reward_sum += reward

  ep_rewards.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

  if done: # an episode finished
    episode_number += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    comb_ep_observations = np.vstack(ep_observations)
    comb_ep_hidden_layer_values = np.vstack(ep_hidden_layer_values)
    comb_ep_gradient_log_ps = np.vstack(ep_gradient_log_ps)
    comb_ep_rewards = np.vstack(ep_rewards)
    ep_observations,ep_hidden_layer_values,ep_gradient_log_ps,ep_rewards = [],[],[],[] # reset array memory

    # compute the discounted reward backwards through time
    discounted_comb_ep_rewards = discount_rewards(comb_ep_rewards) # comb_ep_rewards is an array of the rewards
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_comb_ep_rewards -= np.mean(discounted_comb_ep_rewards) # calculates mean and subtracts from each value
    discounted_comb_ep_rewards /= np.std(discounted_comb_ep_rewards)

    comb_ep_gradient_log_ps *= discounted_comb_ep_rewards # modulate the gradient with advantage (PG magic happens right here.)
    grad = policy_backward(comb_ep_hidden_layer_values, comb_ep_gradient_log_ps)
    for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

    # perform rmsprop parameter update every batch_size episodes
    if episode_number % batch_size == 0:
      for k,v in model.items():
        g = grad_buffer[k] # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print ('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
    if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
    reward_sum = 0
    observation = env.reset() # reset env
    prev_frame = None

  if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
    print('ep %d: game finished, reward: %f' %
              (episode_number, reward) + ('' if reward == -1 else ' !!!!!!!!'))


# TODO: 
# 1. understand logp in policy_foward
# 2. understand dh and backpropogation functions in policy_backward
# 3. understand fake label and variable "y" on 100 & 101 
# 4. understand what the variable "ep_gradient_log_ps" holds & rename if necessary (line 77)
# and change comb_ep_gradient_log_ps accordingly
# 5. understand grad_buffer 
# 6. understand RMSprop calculations and rename variables if necessary 