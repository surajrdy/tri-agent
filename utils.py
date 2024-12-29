# utils.py
import yaml
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_plan(plan_text: str) -> List[Dict]:
    """Parse LLM generated plan text into structured format"""
    # Implementation to parse plan text into action dictionaries
    parsed_actions = []
    # Add parsing logic here
    return parsed_actions

def parse_feedback(feedback_text: str) -> str:
    """Parse LLM generated feedback into structured format"""
    # Implementation to parse feedback text
    # Add parsing logic here
    return feedback_text

def construct_prompt(task_desc: str, 
                    current_plan: List[Dict] = None,
                    feedback: str = None) -> str:
    """Construct prompt for LLM"""
    prompt = f"Task: {task_desc}\n\n"
    if current_plan:
        prompt += f"Current plan:\n{current_plan}\n\n"
    if feedback:
        prompt += f"Feedback:\n{feedback}\n\n"
    return prompt
