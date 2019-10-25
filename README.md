# ReinforceR

Reinforcement Q and Deep Learning agent for automatic control of CartPole model.

<img src="design/reinforcer.jpg?raw=true">

## Detailed
Reinforcement Learning models as application of Deep Learning in the context of Machine Learning using Keras/TensorFlow. Two implementations are included in the package: Deep Neural Networks and Q Learning algorithm. The the neural network is implemented using low level TensorFlow API instead of using higher level perceptrons and learners. As for the second learning method, Q Matrix is at the center of the learning process. Each step the algorithm chooses between the exploting the safe existing knowledge of the best action and exploration of new possibile actions. As a learning platform I have used the Cartpole simulator provided by gym. Please note the model uses API calls specific for TensorFlow v1.14. With the release of the TF2.0 I will update the model.


<img src="design/logo.png?raw=true">

## Demo

Deep Reinforcement Learning model simulation

<img src="design/demo.gif?raw=true">

## Progress

Please note the big drops in reward during the later phases, but these do not influence the mean over 100 episodes much.

<img src="design/learning_progress.png?raw=true">


## Package
- Deep Learning Network
- Q Learning Algorithm
- CUDA Test

## Use CUDA
If you happen to have an Nvidia GPU you could greatly benefit from CUDA speedup.

## How to Contribute

1. Clone repo and create a new branch: `$ git checkout https://github.com/DumitruHanciu/ReinforceR -b new_branch`.
2. Make changes and test
3. Submit Pull Request with comprehensive description of changes

## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**

