import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from gym_game2048.envs.env import Game2048Env

class QRDQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, n_quantiles=200, hidden_size=256):
        super(QRDQNNetwork, self).__init__()
        self.n_quantiles = n_quantiles
        
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Quantile layers - one for each action
        self.quantiles = nn.Linear(hidden_size, action_size * n_quantiles)
        
    def forward(self, state):
        batch_size = state.size(0)
        features = self.features(state)
        quantiles = self.quantiles(features)
        
        # Reshape to (batch_size, action_size, n_quantiles)
        return quantiles.view(batch_size, -1, self.n_quantiles)

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done))
    
    def __len__(self):
        return len(self.buffer)

class QRDQN:
    def __init__(self, state_size, action_size, memory_size=100000, batch_size=128, 
                 learning_rate=0.0001, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.n_quantiles = 200
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = batch_size
        
        # Networks
        self.policy_net = QRDQNNetwork(state_size, action_size).to(device)
        self.target_net = QRDQNNetwork(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer with specified learning rate
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Memory with specified size
        self.memory = ReplayBuffer(capacity=memory_size)
        
        # Quantile midpoints for loss calculation
        self.quantile_tau = torch.FloatTensor(
            (2 * np.arange(self.n_quantiles) + 1) / (2.0 * self.n_quantiles)
        ).to(device)
        
    def huber_loss(self, x, k=1.0):
        return torch.where(x.abs() <= k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))
    
    def quantile_huber_loss(self, predicted, target):
        diff = target.unsqueeze(-1) - predicted.unsqueeze(-2)
        loss = self.huber_loss(diff) * (self.quantile_tau - (diff < 0).float()).abs()
        return loss.mean(dim=-1).sum(dim=-1).mean()
    
    def select_action(self, state, epsilon=0.05):
        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                quantiles = self.policy_net(state)
                action_values = quantiles.mean(dim=2)
                return action_values.max(1)[1].item()
        else:
            return random.randrange(self.action_size)
    
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get current Q values
        current_quantiles = self.policy_net(states)
        current_quantiles = current_quantiles[range(self.batch_size), actions]
        
        # Get next Q values
        with torch.no_grad():
            next_quantiles = self.target_net(next_states)
            best_actions = next_quantiles.mean(dim=2).max(1)[1]
            next_quantiles = next_quantiles[range(self.batch_size), best_actions]
            target_quantiles = rewards.unsqueeze(1) + self.gamma * (1 - dones.unsqueeze(1)) * next_quantiles
        
        # Compute loss
        loss = self.quantile_huber_loss(current_quantiles, target_quantiles)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft update target network
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)
        
        return loss.item()