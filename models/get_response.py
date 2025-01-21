import re

from models.models import *

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
    else:
        print('This method of getting responses is not yet supported!\n')
        return []

def washing_response_4_world_model(response: str) -> str:
    
    # 如果模型调用没有返回结果，直接返回空字符串
    if not response:
        print("模型调用没有返回结果!")
        return ''
        
    # 如果前缀不在response中，说明没有遵循指令，直接返回空字符串
    if prefix_string_world not in response:
        print("前缀不在回复中!")
        return ''
    else:
        response = response.split(prefix_string_world)[-1]
    
    # 将```<content>```中的content提取出来
    state_pattern = re.compile(r"```(.*?)```", re.DOTALL)
    match = re.search(state_pattern, response)
    
    if match:
        response = match.group(1)
    else:
        print("content内容不存在!")
        return ''
    
    return response

def washing_action_4_policy_model(response: str) -> str:
    
    # 如果模型调用没有返回结果，直接返回空字符串
    if not response:
        print("模型调用没有返回结果!")
        return ''
    
    # 如果前缀不在response中，说明没有遵循指令，直接返回空字符串
    if prefix_string_policy not in response:
        print("前缀不在回复中!")
        return ''
    else:
        # find the first occurence of action
        pattern = rf"```((.|\n)*?)```"
        match = re.search(pattern, response)
        if match:
            action = match.group(1).strip()
        else:
            print("content内容不存在!")
            return ''
    
    return response.split(action)[0] + action + "```"

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
        reason_match = re.search(r"Reason:(.*?)\n\n", response, re.DOTALL)
        if reason_match:
            reason_content = reason_match.group(1).strip()
        else:
            print("无理由返回!")
            return '', low
        
        score_match = re.search(r"Score:(.*?)$", response)
        if score_match:
            score_content = score_match.group(1).strip()
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

