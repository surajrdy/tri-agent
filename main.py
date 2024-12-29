# main.py
import sys
sys.path.append('/alfworld')
import os
from planner_agent import PlannerAgent  
from tester_agent import TesterAgent
from optimizer_agent import OptimizerAgent
from alfworld.agents import Environment
from utils import load_config

class EmbodiedPlanningSystem:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        
        # Initialize the three agents
        self.planner = PlannerAgent(
            model_name=self.config.planner.model,
            temperature=self.config.planner.temperature
        )
        
        self.tester = TesterAgent(
            model_name=self.config.tester.model,
            temperature=self.config.tester.temperature
        )
        
        self.optimizer = OptimizerAgent(
            model_name=self.config.optimizer.model,
            temperature=self.config.optimizer.temperature
        )
        
        # Initialize ALFWorld environment
        self.env = Environment(
            split="train", 
            data_path=self.config.env.data_path
        )

    def plan_and_execute(self, task_desc):
        """Main planning and execution loop"""
        max_iterations = self.config.max_iterations
        current_plan = None
        
        for i in range(max_iterations):
            # Step 1: Generate initial plan or get refined plan
            if current_plan is None:
                current_plan = self.planner.generate_plan(task_desc)
            
            # Step 2: Test plan for failure modes
            failure_modes = self.tester.test_plan(
                task_desc, 
                current_plan,
                self.env
            )
            
            # Step 3: If no failures found, execute plan
            if not failure_modes:
                success = self.execute_plan(current_plan)
                if success:
                    return current_plan
                    
            # Step 4: Get optimization feedback
            feedback = self.optimizer.generate_feedback(
                task_desc,
                current_plan, 
                failure_modes
            )
            
            # Step 5: Refine plan
            current_plan = self.planner.refine_plan(
                task_desc,
                current_plan,
                feedback
            )
            
        return current_plan

    def execute_plan(self, plan):
        """Execute plan in ALFWorld environment"""
        obs = self.env.reset()
        done = False
        
        for action in plan:
            obs, reward, done, info = self.env.step(action)
            if done:
                break
                
        return done and reward > 0
