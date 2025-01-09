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
        
        
        ) -> None:
        super().__init__(data, policy_method, reward_method, world_method)
        
        self.init_state = state
        self.branch = branch
        self.roll_policy = roll_policy
        self.roll_branch = roll_branch
        self.roll_forward_steps = roll_forward_steps

        self.time_limit = time_limit
        self.iteration_limit = iteration_limit
        
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
    
    def run(self):
        self.clear_cache()
        self.set_limit_type()
        node, finish, root = MCTS(self)     # input mcts_task
    