import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from gym_game2048.envs.env import Game2048Env

class QRDQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, n_quantiles=200):
        super(QRDQNNetwork, self).__init__()
        self.n_quantiles = n_quantiles
        
        # CNN layers for 4x4 board
        self.conv_layers = nn.Sequential(
            # First layer: 1 -> 32 channels
            nn.Conv2d(1, 32, kernel_size=2, stride=1),  # Output: 32 x 3 x 3
            nn.ReLU(),
            
            # Second layer: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=2, stride=1),  # Output: 64 x 2 x 2
            nn.ReLU(),
            
            # Third layer: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=1, stride=1),  # Output: 128 x 2 x 2
            nn.ReLU(),
            
            # Fourth layer: 128 -> 256 channels
            nn.Conv2d(128, 256, kernel_size=1, stride=1),  # Output: 256 x 2 x 2
            nn.ReLU(),
            
            nn.Flatten()  # Output: 256 * 2 * 2 = 1024
        )
        
        # Fully connected layers with correct input size
        self.fc_layers = nn.Sequential(
            nn.Linear(1024, 512),  # 1024 = 256 channels * 2 * 2
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Dueling architecture
        self.advantage = nn.Linear(256, action_size * n_quantiles)
        self.value = nn.Linear(256, n_quantiles)
    
    def forward(self, state):
        batch_size = state.size(0)
        
        # Reshape state for CNN: (batch_size, 1, 4, 4)
        x = state.view(batch_size, 1, 4, 4)
        
        # Process through CNN
        x = self.conv_layers(x)
        
        # Process through FC layers
        features = self.fc_layers(x)
        
        # Dueling architecture
        advantage = self.advantage(features)
        value = self.value(features)
        
        # Reshape
        advantage = advantage.view(batch_size, -1, self.n_quantiles)
        value = value.view(batch_size, 1, self.n_quantiles)
        
        # Combine value and advantage
        quantiles = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return quantiles
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

class QRDQN:
    def __init__(self, state_size, action_size, memory_size=50000, batch_size=256):
        self.state_size = 37  # Hardcoded to match saved model's input size
        self.action_size = action_size
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_quantiles = 200
        self.gamma = 0.99
        
        # Initialize networks with correct state size
        self.policy_net = QRDQNNetwork(self.state_size, action_size, self.n_quantiles).to(self.device)
        self.target_net = QRDQNNetwork(self.state_size, action_size, self.n_quantiles).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayBuffer(memory_size)  # Use ReplayBuffer instead of deque
        
        self.quantile_tau = torch.FloatTensor([(2 * i + 1) / (2 * self.n_quantiles) for i in range(self.n_quantiles)]).to(self.device)
    def huber_loss(self, x, k=1.0):
        return torch.where(x.abs() <= k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))
    
    def quantile_huber_loss(self, predicted, target):
        diff = target.unsqueeze(-1) - predicted.unsqueeze(-2)
        loss = self.huber_loss(diff) * (self.quantile_tau - (diff < 0).float()).abs()
        return loss.mean(dim=-1).sum(dim=-1).mean()
    
    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            # Preprocess state
            processed_state = self.preprocess_state(state)
            state_tensor = torch.FloatTensor(processed_state).unsqueeze(0).to(self.device)
            
            quantiles = self.policy_net(state_tensor)
            action_values = quantiles.mean(dim=2)
            return action_values.max(1)[1].item()
    
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Preprocess states
        states = np.array([self.preprocess_state(s) for s in states])
        next_states = np.array([self.preprocess_state(s) for s in next_states])
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_quantiles = self.policy_net(states)
        current_quantiles = current_quantiles[range(self.batch_size), actions]
        
        # Next Q values
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
        
        return loss.item()
            

    def preprocess_state(self, state):
        """Simple preprocessing for CNN input"""
        # Convert to numpy array if it's not already
        state = np.array(state, dtype=np.float32)
        
        # Log scale for tile values
        state = np.log2(state + 1)
        
        # Normalize
        state = state / 11.0  # Max possible value is 2048 (2^11)
        
        return state

    def get_dynamic_epsilon(self, max_tile, base_epsilon):
        """Simplified epsilon strategy for quick training"""
        if max_tile >= 128:
            return 0.1
        elif max_tile >= 64:
            return 0.2
        return 0.3  # Higher base epsilon for more exploration