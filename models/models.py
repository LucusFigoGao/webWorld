from openai import AsyncOpenAI

# -------------------- DeepSeek API -------------------- 
API_KEY_2024_12_18 = "<your-api-key>"
API_KEY_2025_01_20 = "<your-api-key>"

# ---------------------- Qwen API ----------------------
API_KEY_2025_01_31 = "<your-api-key>"

completion_tokens = prompt_tokens = 0

deepseek_client = AsyncOpenAI(
    api_key=API_KEY_2024_12_18, base_url="https://api.deepseek.com"
)
qwen_client = AsyncOpenAI(
    api_key=API_KEY_2025_01_31, 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


async def deepseek(messages, model='deepseek-chat', temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    out = []
    cnt = 5
    while cnt:
        try:
            out = (await deepseek_call(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop))[0]
            break
        except Exception as e:
            print(f"Error occurred when getting deepseek reply!\nError type:{e}\n")
            cnt -= 1
    deepseek_usage(backend=model)
    return out

async def qwen(messages, model='qwen-plus', temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    out = []
    cnt = 5
    while cnt:
        try:
            out = (await qwen_call(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop))[0]
            break
        except Exception as e:
            print(f"Error occurred when getting qwen reply!\nError type:{e}\n")
            cnt -= 1
    qwen_usage(backend=model)
    return out

async def deepseek_call(messages, model='deepseek-chat', temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = await deepseek_client.chat.completions.create(
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

async def qwen_call(messages, model='qwen-plus', temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = await qwen_client.chat.completions.create(
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
