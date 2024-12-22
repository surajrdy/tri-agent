import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import math

class RobustEnvironment:
    def __init__(self, state_dim=8, action_dim=8, randomization_range=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.randomization_range = randomization_range
        self.uncertainty_set = {
            'friction': (0.8, 1.2),
            'mass': (0.8, 1.2),
            'force': (0.8, 1.2),
            'noise': (0.0, 0.1)
        }
    
    def randomize_params(self):
        return {k: np.random.uniform(v[0], v[1]) 
                for k, v in self.uncertainty_set.items()}
    
    def step(self, state, action, params):
        if action.size(0) != state.size(0):
            action = action.expand(state.size(0), -1)
            
        noise = torch.randn_like(state) * params['noise']
        next_state = (state + action * params['force'] / params['mass'] + noise) * params['friction']
        reward = -torch.norm(next_state)
        done = torch.norm(next_state) > 10
        return next_state, reward, done
    
    def reset(self):
        return torch.zeros(self.state_dim)

class DirectorAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, action_dim * 2)
        )
    
    def forward(self, state):
        output = self.network(state)
        mean, log_std = output.chunk(2, dim=-1)
        return Normal(mean, log_std.exp())
    
    def robust_objective(self, state, uncertainty_set):
        distributions = [self.forward(state) for _ in range(10)]
        mean_action = torch.stack([d.mean for d in distributions]).mean(0)
        std_action = torch.stack([d.stddev for d in distributions]).mean(0)
        return Normal(mean_action, std_action)

class SocialLearningAgent(nn.Module):
    def __init__(self, state_dim, num_agents):
        super().__init__()
        self.state_dim = state_dim
        self.attention = nn.MultiheadAttention(state_dim, num_heads=4, batch_first=True)
        self.mask_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, num_agents),
            nn.Sigmoid()
        )
        self.history_buffer = []
    
    def forward(self, state, other_states):
        if state.dim() == 1:
            state = state.unsqueeze(0).unsqueeze(0)
        elif state.dim() == 2:
            state = state.unsqueeze(0)
        
        if other_states.dim() == 2:
            other_states = other_states.unsqueeze(0)
        
        masks = self.mask_network(state.squeeze(0))
        
        attended, _ = self.attention(
            state,
            other_states,
            other_states
        )
        
        return attended.squeeze(0), masks.detach()

class ExecutionAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1)
        )
        
    def forward(self, state):
        return self.actor(state)

class TriAgentTrainer:
    def __init__(self, env, director, social, executor):
        self.env = env
        self.director = director
        self.social = social
        self.executor = executor
        self.metrics = {
            'rewards': [],
            'attention_masks': [],
            'robustness_scores': [],
            'uncertainty_metrics': []
        }
    
    def train_episode(self, num_steps=1000):
        state = self.env.reset()
        episode_rewards = []
        attention_masks = []
        uncertainty_metrics = []
        
        for step in range(num_steps):
            params_samples = [self.env.randomize_params() for _ in range(5)]
            director_dist = self.director.robust_objective(state, params_samples)
            director_action = director_dist.sample()
            
            other_states = torch.randn(3, self.env.state_dim)
            social_state, masks = self.social(state, other_states)
            
            mask_np = masks.detach().cpu().numpy()
            if len(mask_np.shape) > 1:
                mask_np = mask_np.reshape(-1)
            
            rewards_under_uncertainty = []
            for params in params_samples:
                final_action = self.executor(social_state)
                _, reward, _ = self.env.step(state, final_action, params)
                rewards_under_uncertainty.append(reward.item())
            
            worst_reward = min(rewards_under_uncertainty)
            episode_rewards.append(worst_reward)
            attention_masks.append(mask_np)
            uncertainty_metrics.append(np.std(rewards_under_uncertainty))
            
            avg_params = {k: np.mean([p[k] for p in params_samples]) 
                        for k in params_samples[0].keys()}
            next_state, _, done = self.env.step(state, final_action, avg_params)
            
            if done:
                break
            state = next_state
        
        max_len = max(len(mask) if isinstance(mask, np.ndarray) else 0 for mask in attention_masks)
        attention_masks = np.array([np.pad(mask, (0, max_len - len(mask)), mode='constant') if len(mask) < max_len else mask for mask in attention_masks])
        
        self.metrics['rewards'].append(np.mean(episode_rewards))
        self.metrics['attention_masks'].append(np.mean(attention_masks, axis=0))
        self.metrics['robustness_scores'].append(np.std(episode_rewards))
        self.metrics['uncertainty_metrics'].append(np.mean(uncertainty_metrics))
        
        return np.mean(episode_rewards)

    def visualize_final_metrics(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0,0].plot(self.metrics['rewards'])
        axes[0,0].set_title('Episode Rewards')
        
        attention_matrix = np.array(self.metrics['attention_masks'])
        sns.heatmap(attention_matrix.T, ax=axes[0,1], cmap='viridis')
        axes[0,1].set_title('Attention Mask Evolution')
        
        axes[1,0].plot(self.metrics['robustness_scores'])
        axes[1,0].set_title('Robustness Scores')
        
        axes[1,1].plot(self.metrics['uncertainty_metrics'])
        axes[1,1].set_title('Uncertainty Impact')
        
        plt.tight_layout()
        plt.show()

def main():
    env = RobustEnvironment()
    director = DirectorAgent(env.state_dim, env.action_dim)
    social = SocialLearningAgent(env.state_dim, num_agents=3)
    executor = ExecutionAgent(env.state_dim, env.action_dim)
    
    trainer = TriAgentTrainer(env, director, social, executor)
    
    num_episodes = 100
    for episode in range(num_episodes):
        reward = trainer.train_episode()
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {reward:.3f}")
    trainer.visualize_final_metrics()

if __name__ == "__main__":
    main()
