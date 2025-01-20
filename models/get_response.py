import re

from models.models import *

prefix_string_world = "In summary, the next web page observation is "


def get_proposal():
    pass

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

def get_value():
    pass

def washing_response_4_world_model(response):
    
    # 如果模型调用没有返回结果，直接返回空字符串
    if not response:
        return ''
        
    # 如果前缀不在response中，说明没有遵循指令，直接返回空字符串
    if prefix_string_world not in response:
        return ''
    else:
        response = response.split(prefix_string_world)[-1]
    
    # 将```<content>```中的content提取出来
    state_pattern = re.compile(r"```(.*?)```", re.DOTALL)
    match = re.search(state_pattern, response)
    
    if match:
        response = match.group(1)
    else:
        return ''
    
    return response


