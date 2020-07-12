import gym
import random
import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

cuda = torch.device("cpu")
#cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Useful for saving state-action transitions in ReplayMemory.memory
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# Memory of the DRL Agent, primarily is a list of state-action transitions
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity  # highest length of self.memory
        self.memory = []  # list of state-action transitions
        self.position = 0  # counter for replacing old memories with new ones

    def push(self, *args):
        # function for adding new memories to the ReplayMemory
        if len(self.memory) < self.capacity:
            self.memory.append(None)  # used for filling out the memory when the agent first starts training
        self.memory[self.position] = Transition(*args)  # adds a Transition namedtuple to the memory
        self.position = (self.position + 1) % self.capacity  # replaces old memories with new ones

    def sample(self, batch_size):
        # returns a bunch of random memories as a Transition tuple consisting of each element of the batch as a tuple
        # so if the batch returned (s1, a1, sn1, r1) and (s2, a1, sn1, r1), this function returns
        #   ( (s1, s2), (a1, a2), (sn1, sn2), (r1, r2) )
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)


# The neural net, this is where the linear transformations are stored
class DQN(nn.Module):
    def __init__(self, numInputs, numOutputs):
        super().__init__()
        self.lin1 = nn.Linear(numInputs, 128)
        self.lin2 = nn.Linear(128, numOutputs)

        # initialize with random weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    # put a tensor through the neural net (this is what is called when you do DQN(input)
    def forward(self, x):
        x = F.leaky_relu(self.lin1(x))
        x = self.lin2(x)
        return x

    # return an action after being given an observation
    def get_action(self, observation, epsilon, env):
        if random.random() > epsilon:
            qvalues = self.forward(observation)
            _, action = torch.max(qvalues, 1)
            return action.numpy()[0]
        else:
            return env.action_space.sample()

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, gamma, batch):
        states = torch.stack(batch.state)
        next_states = torch.stack(batch.next_state)
        actions = torch.Tensor(batch.action).float()
        rewards = torch.Tensor(batch.reward)

        pred = online_net(states).squeeze(1)  # 32x2 tensor, list of qvalues resulting from the states being input
        next_pred = target_net(next_states).squeeze(1)  # into the nets

        pred = torch.sum(pred.mul(actions), dim=1)  # 32 tensor, only keeps the qvalues of the actions in batch.action

        target = rewards + gamma * next_pred.max(1)[0]  # calculates q function of actions

        loss = F.mse_loss(pred, target.detach())  # .detach() is the tensor without the gradient
        optimizer.zero_grad()  # get rid of the grads from the last loop through
        loss.backward()  # calculate gradient of how far off
        optimizer.step()  # some pytorch magic uses the info above to move the optimizer a step in a direction

        return loss


# other useful functions
# replaces one net with the other
def update_target_model(online_net, target_net):
    target_net.load_state_dict(online_net.state_dict())
