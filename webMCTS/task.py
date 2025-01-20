from models.get_response import *
from webMCTS.mcts import MCTS


class SearchTask(object):
    def __init__(self, data, policy_method, reward_method, world_method) -> None:
        """
            :: data: str, represent the question or task or instruction
            :: policy_method: str, default: ['gpt', 'deepseek', 'qwen2.5']
            :: reward_method: str, default: ['gpt', 'deepseek', 'qwen2.5']
            :: world_method: str, default: ['gpt', 'deepseek', 'qwen2.5']
            :: value_cache: dict{str: float}, ['action': value]
        """
        super().__init__()
        self.question = data
        self.policy_method = policy_method
        self.reward_method = reward_method
        self.world_method = world_method
        self.value_cache = {}
    
    def clear_cache(self):
        self.value_cache = {}
    
    @staticmethod
    def get_next_action_prompt_wrap(intent: str, trace: str, state: str, mode: str = "chat") -> str:
        
        from webMCTS.prompt import webarena_cot_id_actrees2str_no_na_prompt as prompt_template
        
        intro = prompt_template["intro"]               # prompt intro
        examples = prompt_template["examples"]         # few examples
        inputs = prompt_template["inputs"]             # input template
        
        current = inputs.format(
            objective=intent,                               # instruction
            observation=state,                              # current web state
            previous_action=trace,                          # action trace
        )

        if mode == "chat":
            """
                [
                    {"role": "system", "content": intro}, 
                    {"role": "system", "name": "example_user", "content": example_input}, 
                    {"role": "system", "name": "example_assistant", "content": example_output}, 
                    {"role": "user", "content": current}
                ]
                >>> {"role": "system", "content": output}
            """
            message = [{"role": "system", "content": intro}]
            for (x, y) in examples:
                message.append(
                    {
                        "role": "system",
                        "name": "example_user",
                        "content": x,
                    }
                )
                message.append(
                    {
                        "role": "system",
                        "name": "example_assistant",
                        "content": y,
                    }
                )
            message.append({"role": "user", "content": current})
        
        elif mode == "completion":
            message = f"{intro}\n\n"
            message += "Here are a few examples:\n"
            for example in examples:
                message += f"Observation\n:{example[0]}\n\n"
                message += f"Action: {example[1]}\n\n"
            message += "Now make prediction given the observation\n\n"
            message += f"Observation\n:{current}\n\n"
            message += "Action:"
        
        return message

    @staticmethod
    def get_next_state_predict_prompt_wrap(state: str, action: str, mode: str = "chat") -> str:
        
        from webMCTS.prompt import world_model_next_state_prediction_prompt as prompt_template
        
        intro = prompt_template["intro"]               # prompt intro
        inputs = prompt_template["inputs"]             # input template
        
        current = inputs.format(
            observation=f"```{state}```",                   # current web state
            action=f"{action}",                             # action trace
        )
        
        if mode == "chat":
            message = [
                {"role": "system", "content": intro}, 
                {"role": "user", "content": current}
            ]
        elif mode == "completion":
            message = f"{intro}\n\n"
            message += f"The current web page observation: ```{state}```\n\n"
            message += f"The previous action: {action}"
        return message
    
    @staticmethod
    def get_step_value_prompt_wrap():
        pass


class MCTS_Task(SearchTask):
    def __init__(
        self, 
        data: str,                                  # str, represent the question or task or instruction
        state: str,                                 # str, represent the initial state of web page
        policy_method,                              # str, default: ['gpt', 'deepseek', 'qwen2.5']
        reward_method,                              # str, default: ['gpt', 'deepseek', 'qwen2.5']
        world_method,                               # str, default: ['gpt', 'deepseek', 'qwen2.5']
        branch=3,                                   # int, the number of sampling times in extension stage
        roll_policy='greedy',                       # str, rollout policy
        roll_branch=1,                              # int, the number of sampling times in rollout stage
        roll_forward_steps=3,                       # int, the number of rollout steps
        time_limit=None,                            # int, time searching limit
        iteration_limit=None,                       # int, iteration searching limit
        reward_model_type='vm',                     # str, reward model type (outcome reward)
        end_gate=0.9,                               # int, threshold of task finished reward
        exploration_constant=0.7,                   # float, MCTS UCB epsilon
        inf=1.0,                                    # float, MCTS UCB avoid stackflow
        low=0,                                      # float
        alpha=0.5,                                  # float, MCTS node value weights
        
        ) -> None:
        super().__init__(data, policy_method, reward_method, world_method)
        
        self.init_state = state
        self.branch = branch
        self.roll_policy = roll_policy
        self.roll_branch = roll_branch
        self.roll_forward_steps = roll_forward_steps

        self.time_limit = time_limit
        self.iteration_limit = iteration_limit
        
        self.end_gate = end_gate
        self.reward_model_type = reward_model_type
        
        self.low = low
        self.INF = inf
        self.alpha = alpha
        self.exploration_constant = exploration_constant
        
    def clear_cache(self):
        self.value_cache = {}
        self.node_count = 1
    
    def set_limit_type(self):
        if self.time_limit is not None:
            if self.iteration_limit is not None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.limit_type = 'time'
        else:
            if self.iteration_limit is None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if self.iteration_limit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.limit_type = 'iterations'
    
    def get_next_action(self, trace, state):
        """
            output:
                >>> child.action = Policy_model(node.trace, node.state)
            input: 
                :: node.trace: action history till current node
                :: node.state: current state of web page
        """
        prompt = self.get_next_action_prompt_wrap(self.question, trace, state, mode="chat")
        response = get_proposal(prompt, self.policy_method)
        response = washing_action_4_policy_model(response)
        return response
    
    def get_next_state_predict(self, state, action):
        """
            output:
                >>> child.state = World_model(node.state, child.action)
            input: 
                :: node.state: current state of web page
                :: child.action: current action from node to child
        """
        prompt = self.get_next_state_predict_prompt_wrap(state, action, mode="chat")
        response = get_state(prompt, self.world_method)
        response = washing_response_4_world_model(response)
        return response
    
    def get_step_value(self, trace, state):
        """
            output:
                >>> child.state = Reward_model(child.trace, child.state)
            input: 
                :: child.trace: action history till child
                :: child.state: next state of web page
        """
        prompt = self.get_step_value_prompt_wrap(trace, state, mode="chat")
        response = get_value(prompt)
        return response
    
    def run(self):
        self.clear_cache()
        self.set_limit_type()
        node, finish, root = MCTS(self)     # input mcts_task
