from openai import OpenAI


API_KEY_2024_12_18 = "sk-3579ef2fa1ab44d6a6ab2335796b10e7"
API_KEY_2025_01_20 = "sk-8059fabdb29a4a09a260e8c0158512e5"


completion_tokens = prompt_tokens = 0

deepseek_client = OpenAI(api_key=API_KEY_2024_12_18, base_url="https://api.deepseek.com")


def deepseek(messages, model='deepseek-chat', temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    out = []
    cnt = 5
    while cnt:
        try:
            out = deepseek_call(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)[0]
            break
        except Exception as e:
            print(f"Error occurred when getting gpt reply!\nError type:{e}\n")
            cnt -= 1
    deepseek_usage(backend=model)
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
        # print(f'得到GPT回复:{res}\n\n')
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


def local_inference_model(
    prompt, 
    max_length=2048, 
    truncation=True, 
    do_sample=True,
    max_new_tokens=1024, 
    temperature=0.7
    ):
    pass