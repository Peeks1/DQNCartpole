import gym
import Agent as a
import torch
import numpy as np

# hyperparameters (and other variables)
lr = .001
gamma = .99
batch_size = 32
min_epsilon = .1
update_target = 100
memory_capacity = 1000
training_episodes = 3000
epsilon_decrement = .00005
random_action_period = 1000

# make env
env = gym.make('CartPole-v1')

# get numbers needed for function inputs
numInputs = env.observation_space.shape[0]
numActions = env.action_space.n

# make net variables
online_net = a.DQN(numInputs, numActions)
target_net = a.DQN(numInputs, numActions)
a.update_target_model(online_net, target_net)

# define optimizer
optimizer = torch.optim.Adam(online_net.parameters(), lr=lr)

# put the nets on the GPU and prime them for training
online_net.to(a.cuda)
target_net.to(a.cuda)
online_net.train()
target_net.train()

# other variables that need to be defined outside of the training loop
memory = a.ReplayMemory(memory_capacity)
running_score = 0
epsilon = 1
steps = 0
loss = 0

# the training loop
for e in range(training_episodes):
    done = False
    score = 0
    state = env.reset()
    state = torch.Tensor(state).to(a.cuda)  # env.state returns a np array, so need to transform it
    state = state.unsqueeze(0)  # and this makes it a 1xn tensor

    # the episode loop
    print("Episode", e)
    while not done:
        if e % 100 == 0:
            env.render()

        steps += 1

        action = target_net.get_action(state, epsilon, env)
        next_state, reward, done, _ = env.step(action)

        next_state = torch.Tensor(next_state).to(a.cuda)
        next_state = next_state.unsqueeze(0)

        reward = reward if not done or score == 499 else -1  # conditional is to punish for losing without making to end
        action_one_hot = np.zeros(numActions)
        action_one_hot[action] = 1
        memory.push(state, action_one_hot, next_state, reward)

        score += reward
        state = next_state

        if steps > random_action_period:
            if epsilon > min_epsilon:
                epsilon -= epsilon_decrement

            batch = memory.sample(batch_size)
            loss = a.DQN.train_model(online_net, target_net, optimizer, gamma, batch)

            if steps % update_target == 0:
                a.update_target_model(online_net, target_net)
    print("Score", score)
    running_score = .99 * running_score + .01 * score
    print("Running Score", running_score)




