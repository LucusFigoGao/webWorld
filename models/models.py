import json
import requests
from openai import OpenAI

# -------------------- DeepSeek API -------------------- 
API_KEY_2024_12_18 = "sk-3579ef2fa1ab44d6a6ab2335796b10e7"
API_KEY_2025_01_20 = "sk-8059fabdb29a4a09a260e8c0158512e5"

# ---------------------- Qwen API ----------------------
API_KEY_2025_01_31 = "sk-00540965ccd94b79966b8c419f6ad21a"

# ---------------------- Siliconflow API ----------------------
API_KEY_2025_02_04 = "sk-llvashpiiyxphjvhvozwnpfplxkotpgetbrqbmnwubdqhzvq"


completion_tokens = prompt_tokens = 0

deepseek_client = OpenAI(api_key=API_KEY_2024_12_18, base_url="https://api.deepseek.com")
qwen_client = OpenAI(api_key=API_KEY_2025_01_31, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


def deepseek(messages, model='deepseek-chat', temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    out = []
    cnt = 5
    while cnt:
        try:
            out = deepseek_call(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)[0]
            break
        except Exception as e:
            print(f"Error occurred when getting deepseek reply!\nError type:{e}\n")
            cnt -= 1
    deepseek_usage(backend=model)
    return out

def qwen(messages, model='qwen-plus', temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    out = []
    cnt = 5
    while cnt:
        try:
            out = qwen_call(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)[0]
            break
        except Exception as e:
            print(f"Error occurred when getting qwen reply!\nError type:{e}\n")
            cnt -= 1
    qwen_usage(backend=model)
    return out

def deepseek_call(messages, model='deepseek-chat', temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = deepseek_client.chat.completions.create(
            model=model,
            messages=messages, 
            stream=False, 
            temperature=temperature, 
            max_tokens=max_tokens
        )
        # print(f'得到DeepSeek回复:{res}\n\n')
        outputs.extend([choice.message.content for choice in res.choices])
        # log completion tokens
        completion_tokens += res.usage.completion_tokens
        prompt_tokens += res.usage.prompt_tokens
    return outputs

def qwen_call(messages, model='qwen-plus', temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = qwen_client.chat.completions.create(
            model=model,
            messages=messages, 
            stream=False, 
            temperature=temperature, 
            max_tokens=max_tokens
        )
        # print(f'得到Qwen回复:{res}\n\n')
        outputs.extend([choice.message.content for choice in res.choices])
        # log completion tokens
        completion_tokens += res.usage.completion_tokens
        prompt_tokens += res.usage.prompt_tokens
    return outputs

def siliconflow(messages, model='Qwen/Qwen2.5-72B-Instruct', temperature=0.7, max_tokens=1000, n=1) -> list:
    out = []
    cnt = 5
    while cnt:
        try:
            out = siliconflow_call(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=1)[0]
            break
        except Exception as e:
            print(f"Error occurred when getting siliconflow reply!\nError type:{e}\n")
            cnt -= 1
    siliconflow_usage(backend=model)     
    return out

def siliconflow_call(messages: list, model='Qwen/Qwen2.5-72B-Instruct', temperature=0.7, max_tokens=1000, 
                     top_p=0.7, top_k=50, frequency_penalty=0.5, n=1):
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        response = requests.request(
            "POST", 
            url="https://api.siliconflow.cn/v1/chat/completions", 
            json={
                "model": model,
                "messages": messages, 
                "stream": False,
                "max_tokens": max_tokens,
                "stop": ["null"],
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "frequency_penalty": frequency_penalty,
                "n": n,
                "response_format": {"type": "text"},
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "description": "<string>",
                            "name": "<string>",
                            "parameters": {},
                            "strict": False
                        }
                    }
                ]
            }, 
            headers={
                "Authorization": f"Bearer {API_KEY_2025_02_04}",
                "Content-Type": "application/json"
            }
        )
        res = json.loads(response.text)
        outputs.extend([choice['message']['content'] for choice in res['choices']])
        # log completion tokens
        completion_tokens += res['usage']['completion_tokens']
        prompt_tokens += res['usage']['prompt_tokens']
    return outputs

def deepseek_usage(backend='deepseek-chat'):
    global completion_tokens, prompt_tokens
    if backend == "deepseek-chat":
        cost = completion_tokens / 1000000 * 0.1 + prompt_tokens / 1000000 * 2
    else:
        cost = -1
    print({"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost})
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}

def qwen_usage(backend='qwen-plus'):
    global completion_tokens, prompt_tokens
    if backend == "qwen-plus":
        cost = completion_tokens / 1000000 * 0.8 + prompt_tokens / 1000000 * 2
    else:
        cost = -1
    print({"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost})
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}

def siliconflow_usage(backend):
    global completion_tokens, prompt_tokens
    if backend == "deepseek-ai/DeepSeek-V3":
        cost = completion_tokens / 1000000 * 1 + prompt_tokens / 1000000 * 2
    else:
        cost = -1
    print({"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost})
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}

def local_inference_model(
    prompt, 
    max_length=2048, 
    truncation=True, 
    do_sample=True,
    max_new_tokens=1024, 
    temperature=0.7
    ):
    pass