import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

class RobustEnvironment:
    def __init__(self, state_dim=8, action_dim=4, randomization_range=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.randomization_range = randomization_range
        
    def randomize_params(self):
        return {
            'friction': 1.0 + np.random.uniform(-self.randomization_range, self.randomization_range),
            'mass': 1.0 + np.random.uniform(-self.randomization_range, self.randomization_range),
            'force': 1.0 + np.random.uniform(-self.randomization_range, self.randomization_range)
        }
        
    def step(self, state, action, params):
        # Simulate environment dynamics with randomized parameters
        next_state = state + action * params['force'] / params['mass']
        reward = -torch.norm(next_state)
        done = torch.norm(next_state) > 10
        return next_state, reward, done

class DirectorAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim * 2)  # Mean and std
        )
        
    def forward(self, state):
        output = self.network(state)
        mean, log_std = output.chunk(2, dim=-1)
        return Normal(mean, log_std.exp())

class SocialLearningAgent(nn.Module):
    def __init__(self, state_dim, num_agents):
        super().__init__()
        self.attention = nn.MultiheadAttention(state_dim, num_heads=1)
        self.mask_network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_agents),
            nn.Sigmoid()
        )
        
    def forward(self, state, other_states):
        # Compute inverse attention masks
        masks = self.mask_network(state)
        
        # Apply masked attention
        attended, _ = self.attention(
            state.unsqueeze(0),
            other_states.unsqueeze(0),
            other_states.unsqueeze(0),
            key_padding_mask=(masks > 0.5).unsqueeze(0)
        )
        return attended.squeeze(0), masks

class ExecutionAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
class TriAgentTrainer:
    def __init__(self, env, director, social, executor):
        self.env = env
        self.director = director
        self.social = social
        self.executor = executor
        self.metrics = {
            'rewards': [],
            'attention_masks': [],
            'robustness_scores': []
        }
        
    def train_episode(self, num_steps=1000):
        state = torch.zeros(self.env.state_dim)
        episode_rewards = []
        attention_masks = []
        
        for step in range(num_steps):
            # Domain randomization
            params = self.env.randomize_params()
            
            # Director action
            director_dist = self.director(state)
            director_action = director_dist.sample()
            
            # Social learning with inverse attention
            other_states = torch.randn(3, self.env.state_dim)  # Simulated other agents
            social_state, masks = self.social(state, other_states)
            attention_masks.append(masks.detach().numpy())
            
            # Executor action
            final_action = self.executor.actor(social_state)
            
            # Environment step
            next_state, reward, done = self.env.step(state, final_action, params)
            episode_rewards.append(reward.item())
            
            if done:
                break
            state = next_state
            
        self.metrics['rewards'].append(np.mean(episode_rewards))
        self.metrics['attention_masks'].append(np.mean(attention_masks, axis=0))
        self.metrics['robustness_scores'].append(
            np.std(episode_rewards) / np.mean(episode_rewards)
        )
        
    def visualize_metrics(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot rewards
        axes[0,0].plot(self.metrics['rewards'])
        axes[0,0].set_title('Episode Rewards')
        axes[0,0].set_xlabel('Episode')
        axes[0,0].set_ylabel('Average Reward')
        
        # Plot attention mask evolution
        attention_matrix = np.array(self.metrics['attention_masks'])
        sns.heatmap(attention_matrix.T, ax=axes[0,1], cmap='viridis')
        axes[0,1].set_title('Attention Mask Evolution')
        axes[0,1].set_xlabel('Episode')
        axes[0,1].set_ylabel('Agent')
        
        # Plot robustness scores
        axes[1,0].plot(self.metrics['robustness_scores'])
        axes[1,0].set_title('Robustness Scores')
        axes[1,0].set_xlabel('Episode')
        axes[1,0].set_ylabel('Coefficient of Variation')
        
        # Plot final attention distribution
        sns.boxplot(data=attention_matrix, ax=axes[1,1])
        axes[1,1].set_title('Final Attention Distribution')
        axes[1,1].set_xlabel('Agent')
        axes[1,1].set_ylabel('Attention Weight')
        
        plt.tight_layout()
        plt.show()
def main():
    # Initialize environment and agents
    env = RobustEnvironment()
    director = DirectorAgent(env.state_dim, env.action_dim)
    social = SocialLearningAgent(env.state_dim, num_agents=3)
    executor = ExecutionAgent(env.state_dim, env.action_dim)
    
    # Initialize trainer
    trainer = TriAgentTrainer(env, director, social, executor)
    
    # Training loop
    num_episodes = 100
    for episode in range(num_episodes):
        trainer.train_episode()
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Average Reward: {trainer.metrics['rewards'][-1]:.3f}")
    
    # Visualize results
    trainer.visualize_metrics()

if __name__ == "__main__":
    main()
