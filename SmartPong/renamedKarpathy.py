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
  # print(model['W1'])
  model['W2'] = np.random.randn(H) / np.sqrt(H) #Divide by square root of number of dimension size to normalize weights

grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory
# np.zeros_like - Return an array of zeros with the same shape and type as a given array.

def sigmoid(x): 
    # print(x)
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

def policy_forward(change_in_frame):
  hidden_layer_values = np.dot(model['W1'], change_in_frame)
  print(change_in_frame[3200:])
  #print(h)
  hidden_layer_values[hidden_layer_values<0] = 0 # ReLU nonlinearity
  #print("hidden_layer1_values is: ", hidden_layer1_values)
  logp = np.dot(model['W2'], hidden_layer_values)
  #print(logp)
  output_probability = sigmoid(logp)
  # print(output_probability)
  return output_probability, hidden_layer_values # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  #eph.T and dh.T are the transposed matrices of eph and dh, respectively
  dW2 = np.dot(eph.T, epdlogp).ravel()
  #print("eph is: ", eph)          # What is the difference between eph and eph.T?
  #print("eph.T is:", eph.T)       # We probably have to fully understand the backpropagation equation to understand this function
  #print("epdlogp is:", epdlogp)
  dh = np.outer(epdlogp, model['W2'])
  #print("dh is: ", dh)
  #print("dh.T is: ", dh.T)
  dh[eph <= 0] = 0 # backpro prelu
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}

env = gym.make("Pong-v0")
observation = env.reset() # gets very first image of the game
prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
while True:
  if render: 
    env.render()
    #time.sleep(0.5)

  # preprocess the observation, set input to network to be difference image
  cur_x = prepro(observation)
  change_in_frame = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x

  # forward the policy network and sample an action from the returned probability
  aprob, h = policy_forward(change_in_frame)
  action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

  # record various intermediates (needed later for backprop)
  xs.append(change_in_frame) # observation
  hs.append(h) # hidden state
  y = 1 if action == 2 else 0 # a "fake label"
  dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

  # step the environment and get new measurements
  observation, reward, done, info = env.step(action)
  reward_sum += reward


  drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

  if done: # an episode finished
    episode_number += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    epx = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs)
    xs,hs,dlogps,drs = [],[],[],[] # reset array memory

    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr) # epr is an array of the rewards
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr) # calculates mean and subtracts from each value
    discounted_epr /= np.std(discounted_epr)

    epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
    grad = policy_backward(eph, epdlogp)
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
    prev_x = None

  if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
    print('ep %d: game finished, reward: %f' %
              (episode_number, reward) + ('' if reward == -1 else ' !!!!!!!!'))
              