# planner_agent.py
import openai
from typing import List, Dict

class PlannerAgent:
    def __init__(self, model_name: str, temperature: float = 0.7):
        self.model = model_name
        self.temperature = temperature
        
    def generate_plan(self, task_desc: str) -> List[Dict]:
        """Generate initial plan from task description"""
        prompt = self._construct_planning_prompt(task_desc)
        
        response = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=500
        )
        
        return self._parse_plan(response.choices[0].text)
        
    def refine_plan(self, task_desc: str, 
                    current_plan: List[Dict],
                    feedback: str) -> List[Dict]:
        """Refine plan based on feedback"""
        prompt = self._construct_refinement_prompt(
            task_desc, current_plan, feedback
        )
        
        response = openai.Completion.create(
            model=self.model, 
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=500
        )
        
        return self._parse_plan(response.choices[0].text)

    def _construct_planning_prompt(self, task_desc: str) -> str:
        return f"""Given the following task in ALFWorld:
        {task_desc}
        
        Generate a detailed plan with action sequences that would accomplish this task.
        Format the plan as a list of action dictionaries.
        """

    def _construct_refinement_prompt(self, task_desc: str,
                                   current_plan: List[Dict],
                                   feedback: str) -> str:
        return f"""Task: {task_desc}
        
        Current plan:
        {current_plan}
        
        Feedback from testing:
        {feedback}
        
        Generate an improved plan that addresses the feedback.
        """
        
    def _parse_plan(self, plan_text: str) -> List[Dict]:
        """Parse generated text into structured plan"""
        # Implementation to parse text into action sequence
        pass
