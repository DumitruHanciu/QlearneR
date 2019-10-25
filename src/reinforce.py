import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

num_training_episodes = 20000
episode_length = 200
in_u = 4
small_network = 0
render = 0
use_normalisation = 1

# Network
if (not small_network):
  h_u = 3
  input = tf.placeholder(tf.float32, (None, in_u))
  W1 = tf.Variable(tf.random_normal((in_u, h_u)))
  b1 = tf.Variable(tf.zeros([h_u]))
  h1 = tf.nn.tanh(tf.matmul(input, W1) + b1)
  W2 = tf.Variable(tf.random_normal((h_u, 2),stddev=0.1))
  b2 = tf.Variable(tf.zeros([2]))
  y = tf.matmul(h1, W2) + b2
  pred_act = tf.multinomial(y, 1)

# Alternative Small Network
else:
  input = tf.placeholder(tf.float32, (None, in_u))
  W = tf.Variable(tf.random_normal((in_u, 2),stddev=0.1))
  b = tf.Variable(tf.zeros([2]))
  y = tf.matmul(input, W) + b
  pred_act = tf.multinomial(y, 1)

# Learning rate decay
INITIAL_LEARNING_RATE =  0.002
LEARNING_RATE_DECAY_FACTOR = 1 #0.5
global_step = tf.contrib.framework.get_or_create_global_step()
decay_steps = 500
lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,global_step,decay_steps,LEARNING_RATE_DECAY_FACTOR,staircase=True)

# Training
taken_act = tf.placeholder(tf.int32, (None,))
reward_return = tf.placeholder(tf.float32, (None,))
cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=taken_act)
loss = tf.reduce_mean(cross_entropy_loss)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
gradients = optimizer.compute_gradients(loss)
for i, (grad, var) in enumerate(gradients):
  if grad is not None:
    gradients[i] = (grad * reward_return, var)
train = optimizer.apply_gradients(gradients)

# For normalisation to reduce training variance
all_G = []
max_G_length = 10000

# Run one episode
def run_episode(env, sess):
  observation = env.reset()
  episode_return = 0
  actions = []
  observations = []

  # Run one time step
  for _ in range(episode_length):

    # If not using all of the inputs
    if in_u == 2:
      only_angles = 1
      if not only_angles:
        observation = observation[0:3:2]
      else:
        observation = observation[2:4]

    # Get actions from network
    action = sess.run(pred_act, {input: observation[np.newaxis,:]})[0][0]

    # Save data
    observations.append(observation)
    actions.append(action)

    # Get rewards and observations from environemnt
    observation, reward, done, info = env.step(action)

    episode_return += reward

    # disable rendering for faster training
    if (render):
      env.render()

    if done:
      break

  # Get returns and normalise them for lower variance training
  n = int(episode_return)
  G = np.zeros(n)
  global all_G
  for t in range(n - 1):
    G[t] = n-t
  if use_normalisation:
    all_G += G.tolist()
    all_G = all_G[:max_G_length]
    G -= np.mean(all_G)
    G /= np.std(all_G)

  # Run training for each timestep
  for t in range(n-1):
    g = np.array([G[t]])
    s = observations[t][np.newaxis, :]
    a = np.array([actions[t]])
    sess.run([train], {input:s, taken_act:a, reward_return:g})

  return episode_return

# Setup environment and session
env = gym.make('CartPole-v0')
monitor = gym.wrappers.Monitor(env, 'cartpole/', force=True)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Data for plotting
returns = []
avs = []
plt.ion()
smooth = 100
end_num = 195
prev = np.zeros(smooth)

# Run each episode
for i in range(num_training_episodes):
  episode_return = run_episode(env, sess)

  # Record the data for each episode
  returns.append(episode_return)
  prev[0:-1] = prev[1:]
  prev[-1] = episode_return
  av = np.mean(prev[prev != 0])
  avs.append(av)

  # Plot every 100 episodes
  if i % 100 == 99 or av > end_num:
    plt.plot(returns, c='b')
    plt.plot(avs, c='r')
    plt.title("Return for each episode")
    plt.xlabel("Episode Number")
    plt.ylabel("Return (number of timesteps lasted)")
    plt.legend(["Return", "Mean over last 100"],bbox_to_anchor=(0.88,0.25),bbox_transform=plt.gcf().transFigure)
    plt.pause(1e-20)

  # End if successful
  if (av > end_num):
    print("done@%i" % i)
    break

# Clean up
plt.ioff()
plt.show()
monitor.close()
