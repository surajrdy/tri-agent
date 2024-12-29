# tester_agent.py
import openai
from typing import List, Dict
from alfworld.agents import Environment

class TesterAgent:
    def __init__(self, model_name: str, temperature: float = 0.7):
        self.model = model_name
        self.temperature = temperature

    def test_plan(self, task_desc: str, 
                  plan: List[Dict],
                  env: Environment) -> List[str]:
        """Test plan for potential failure modes"""
        # First use LLM to identify potential failure modes
        potential_failures = self._identify_failure_modes(task_desc, plan)
        
        # Then simulate plan execution to verify failures
        verified_failures = self._verify_failures(
            plan, potential_failures, env
        )
        
        return verified_failures

    def _identify_failure_modes(self, task_desc: str,
                              plan: List[Dict]) -> List[str]:
        """Use LLM to identify potential failure modes"""
        prompt = self._construct_testing_prompt(task_desc, plan)
        
        response = openai.Completion.create(
            model=self.model,
            prompt=prompt, 
            temperature=self.temperature,
            max_tokens=300
        )
        
        return self._parse_failures(response.choices[0].text)

    def _verify_failures(self, plan: List[Dict],
                        potential_failures: List[str],
                        env: Environment) -> List[str]:
        """Verify potential failures through simulation"""
        verified_failures = []
        
        for failure in potential_failures:
            if self._simulate_failure_case(plan, failure, env):
                verified_failures.append(failure)
                
        return verified_failures

    def _simulate_failure_case(self, plan: List[Dict],
                             failure: str,
                             env: Environment) -> bool:
        """Simulate specific failure case in environment"""
        # Reset env with failure case conditions
        obs = env.reset()
        
        for action in plan:
            obs, reward, done, info = env.step(action)
            if done and reward <= 0:
                return True
                
        return False
