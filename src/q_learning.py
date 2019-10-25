# Imports
import gym
import numpy as np
import random
import matplotlib.pyplot as plt

num_training_episodes = 20000
episode_length = 200
render = 0
mode = 2 # 0 = full, 1 = no derivatives, 2 = only angles
use_soft_max_exploration = 0

# Training parameters
if mode == 1: bin_sizes = np.array([10,1,10,1])
elif mode == 2: bin_sizes = np.array([1,1,10,10])
else: bin_sizes = np.array([10,10,10,10])
bounds = [2.4, 1, 0.25, 1]
alpha = 0.3
gamma = 1
if not use_soft_max_exploration:
  eps = 0.5
  min_eps = 0.01
  decay_rate = 0.99999
else:
  if mode == 0:
    eps = 6
  else:
    eps = 12
  min_eps = eps
  decay_rate = 1

# Q matrix
Q = np.random.rand(bin_sizes[0],bin_sizes[1],bin_sizes[2],bin_sizes[3],2)

# Run one episode
def run_episode(env):
  observation = env.reset()
  episode_return = 0
  global eps,alpha

  # Sample from our model
  def sample_model(observation):
    bin = np.floor(bin_sizes / 2 * (observation / bounds + 1)).astype(int)
    bin[bin > bin_sizes - 1] = bin_sizes[bin > bin_sizes - 1] - 1
    bin[bin < 0] = 0
    Qs = Q[bin[0]][bin[1]][bin[2]][bin[3]]
    return Qs, bin

  # Softmax used in exploration strategy
  def softmax(w, t=1.0):
    x = np.array(w)
    e = np.exp((x-np.max(x)) / t)
    dist = e / np.sum(e)
    return dist

  # Run one timestep
  for _ in range(episode_length):

    # Get output from our model
    Qs,bin = sample_model(observation)

    # Get action using specified exploration strategy
    if not use_soft_max_exploration:
      # Epsilon greedy
      if random.uniform(0, 1) < eps:
        action = 0 if random.uniform(0, 1) > 0.5 else 1
      else:
        action = Qs.argsort()[-1]
    else:
      # Softmax
      prob = softmax(Qs,eps)
      action = 0 if random.uniform(0, 1) < prob[0] else 1

    # Get reward and next state from environment
    observation, reward, done, info = env.step(action)

    # Sample our model for the next timestep and get most likely action
    nQs,_ = sample_model(observation)
    n_act = nQs.argsort()[-1]

    # Update model using learning rule
    Q[bin[0]][bin[1]][bin[2]][bin[3]][action] += alpha * (reward + gamma*nQs[n_act] - Qs[action])

    # Decay epsilon
    eps = max(min_eps,eps*decay_rate)

    # Sum return
    episode_return += reward

    # disable rendering for faster training
    if render:
      env.render()

    if done:
      break

  return episode_return

# Set up environment
env = gym.make('CartPole-v0')
monitor = gym.wrappers.Monitor(env, 'cartpole/', force=True)

# Plotting constants
returns = []
epses= []
avs = []
plt.ion()
smooth = 100
end_num = 195
prev = np.zeros(smooth)

# Run each episode
for i in range(num_training_episodes):
  episode_return =  run_episode(env)

  # Record the data for each episode
  returns.append(episode_return)
  prev[0:-1] = prev[1:]
  prev[-1] = episode_return
  av = np.mean(prev[prev != 0])
  avs.append(av)

  # Plots the return every 1000 timesteps or at the end
  if i%1000 == 999 or av>end_num:
    plt.figure(1)
    plt.plot(returns,c='b')
    plt.plot(avs, c='r')
    plt.title("Return for each episode")
    plt.xlabel("Episode Number")
    plt.ylabel("Return (number of timesteps lasted)")
    plt.legend(["Return", "Mean over last 100"], bbox_to_anchor=(0.88, 0.25), bbox_transform=plt.gcf().transFigure)
    plt.pause(1e-20)

  # Ends if successful in environment
  if(av>end_num):
    print("done@%i"%i)
    break

# Clean up
plt.ioff()
plt.show()
monitor.close()
