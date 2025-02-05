import re

from models.models import *
from utils.text_utils import *

prefix_string_world = "In summary, the next web page observation is "
prefix_string_policy = "In summary, the next action I will perform is"


def get_proposal(
    prompt: str, 
    policy_model: str, 
    temperature: float = 0.7, 
    max_tokens: int = 4096, 
    seed: int = 170, 
    max_length: int = 8192, 
    truncation: bool = True,
    do_sample: bool = True, 
    max_new_tokens: int = 4096
    ):
    response = []
    cnt = 2
    
    if policy_model == 'deepseek-chat':
        while not response and cnt:
            response = deepseek(prompt, model=policy_model, temperature=temperature, max_tokens=max_tokens)
            cnt -= 1
        if not response:
            print(f'obtain<{policy_model}>response fail!\n')
            return []
        else:
            return response
    elif 'qwen' in policy_model:
        while not response and cnt:
            response = qwen(prompt, model=policy_model, temperature=temperature, max_tokens=max_tokens)
            cnt -= 1
        if not response:
            print(f'obtain<{policy_model}>response fail!\n')
            return []
        else:
            return response
    elif policy_model == 'Qwen/Qwen2.5-72B-Instruct':
        while not response and cnt:
            response = siliconflow(prompt, model=policy_model, temperature=temperature, max_tokens=max_tokens)
            cnt -= 1
        if not response:
            print(f'obtain<{policy_model}>response fail!\n')
            return []
        else:
            return response
    else:
        print('This method of getting responses is not yet supported!\n')
        return []

def get_state(
    prompt: str, 
    world_method: str, 
    temperature: float = 0.7, 
    max_tokens: int = 4096, 
    seed: int = 170, 
    max_length: int = 8192, 
    truncation: bool = True,
    do_sample: bool = True, 
    max_new_tokens: int = 4096
    ):
    response = []
    cnt = 2
    
    if world_method == 'deepseek-chat':
        while not response and cnt:
            response = deepseek(prompt, model=world_method, temperature=temperature, max_tokens=max_tokens)
            cnt -= 1
        if not response:
            print(f'obtain<{world_method}>response fail!\n')
            return []
        else:
            return response
    
    elif 'qwen' in world_method:
        while not response and cnt:
            response = qwen(prompt, model=world_method, temperature=temperature, max_tokens=max_tokens)
            cnt -= 1
        if not response:
            print(f'obtain<{world_method}>response fail!\n')
            return []
        else:
            return response
    
    elif world_method == 'Qwen/Qwen2.5-72B-Instruct':
        while not response and cnt:
            response = siliconflow(prompt, model=world_method, temperature=temperature, max_tokens=max_tokens)
            cnt -= 1
        if not response:
            print(f'obtain<{world_method}>response fail!\n')
            return []
        else:
            return response
    
    elif world_method == 'local':
        while not response and cnt:
            response = local_inference_model(
                prompt, max_length=max_length, truncation=truncation, do_sample=do_sample,
                max_new_tokens=max_new_tokens, temperature=temperature
            )
            cnt -= 1
        if not response:
            print(f'obtain<{world_method}>response fail!\n')
            return []
        else:
            return response
    
    else:
        print('This method of getting responses is not yet supported!\n')
        return []

def get_value(
    prompt: str, 
    reward_model: str, 
    temperature: float = 0.7, 
    max_tokens: int = 4096, 
    seed: int = 170, 
    max_length: int = 8192, 
    truncation: bool = True,
    do_sample: bool = True, 
    max_new_tokens: int = 4096
    ):
    response = []
    cnt = 2
    
    if reward_model == 'deepseek-chat':
        while not response and cnt:
            response = deepseek(prompt, model=reward_model, temperature=temperature, max_tokens=max_tokens)
            cnt -= 1
        if not response:
            print(f'obtain<{reward_model}>response fail!\n')
            return []
        else:
            return response
    elif 'qwen' in reward_model:
        while not response and cnt:
            response = qwen(prompt, model=reward_model, temperature=temperature, max_tokens=max_tokens)
            cnt -= 1
        if not response:
            print(f'obtain<{reward_model}>response fail!\n')
            return []
        else:
            return response
    elif reward_model == 'Qwen/Qwen2.5-72B-Instruct':
        while not response and cnt:
            response = siliconflow(prompt, model=reward_model, temperature=temperature, max_tokens=max_tokens)
            cnt -= 1
        if not response:
            print(f'obtain<{reward_model}>response fail!\n')
            return []
        else:
            return response
    else:
        print('This method of getting responses is not yet supported!\n')
        return []

def washing_response_4_world_model(response: str) -> str:
    
    # 如果模型调用没有返回结果，直接返回空字符串
    if not response:
        print("模型调用没有返回结果!")
        return ''
    
    # 将```<content>```中的content提取出来
    state_pattern = re.compile(r"```(.*?)```", re.DOTALL)
    match = re.search(state_pattern, response)
    
    if match:
        response = match.group(1)
    else:
        print("content内容不存在!")
        return ''
    
    # 如果前缀不在response中，说明没有遵循指令，直接返回空字符串
    if prefix_string_world not in response:
        print(f"前缀{prefix_string_world}不在回复中!")
    else:
        response = response.split(prefix_string_world)[-1]
    
    return response

def washing_action_4_policy_model(response: str, state: str) -> str:
    
    # 如果模型调用没有返回结果，直接返回空字符串
    if not response:
        print("模型调用没有返回结果!")
        return '', ''
    
    # find the first occurence of action
    pattern = rf"```((.|\n)*?)```"
    match = re.search(pattern, response)
    if match:
        action = match.group(1).strip()
    else:
        print("content(action)内容不存在!")
        return '', ''
    
    # 这里从state中将[id]的具体内容补全
    action_completed = action_completion(action, state)
    if action_completed is None:        # 说明当前action存在问题
        return '', ''
    
    # 如果前缀不在response中，说明没有遵循指令，直接返回空字符串
    if prefix_string_policy not in response:
        print(f"前缀'{prefix_string_policy}'不在回复中!")
        return prefix_string_policy + action_completed, action_completed
    
    return response.split("```"+action+"```")[0] + action_completed, action_completed

def washing_value_4_reward_model(response: str, low=0.0, high=5.0) -> str:
    # 如果模型调用没有返回结果，直接返回空字符串
    if not response:
        print("模型调用没有返回结果!")
        return '', low
    
    # 如果前缀不在response中，说明没有遵循指令，直接返回空字符串
    if "Reason" not in response or "Score" not in response:
        print("前缀不在回复中!")
        return '', low
    else:
        pattern = r"Reason:\s*(.*?)\s*Score:\s*(\d+)"
        match = re.search(pattern, response, re.DOTALL)
        
        if match:
            reason_content = match.group(1).strip()
        else:
            print("无理由返回!")
            reason_content = ""
        
        if match:
            score_content = match.group(2).strip()
            try:
                score = float(score_content)
                score = min(max(low, score), high)
            except Exception as e:
                print(f'分数输出有误！错误类型:{e}\n')
                return reason_content, low
        else:
            print("无分数输出!")
            return reason_content, low
    
    return reason_content, score