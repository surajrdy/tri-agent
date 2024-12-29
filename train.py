# train.py
import torch
from torch.utils.data import DataLoader
from alfworld.agents import Environment
from main import EmbodiedPlanningSystem

def train(config_path="config.yaml"):
    # Initialize system
    system = EmbodiedPlanningSystem(config_path)
    
    # Load ALFWorld training data
    train_env = Environment(split="train")
    train_loader = DataLoader(train_env, batch_size=32)
    
    # Training loop
    for epoch in range(config.epochs):
        for batch in train_loader:
            tasks, _ = batch
            
            # Generate and execute plans for each task
            for task in tasks:
                final_plan = system.plan_and_execute(task)
                
                # Log results and update metrics
                # Implementation for logging and metrics
                
if __name__ == "__main__":
    train()
