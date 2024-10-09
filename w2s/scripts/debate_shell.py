import os

model_root = "/mnt/sdc1/ModelWarehouse"
model_name = ["qwen-7b-chat"]
ROUND = [1, 2, 3, 4]
MASK = [True, False]
# length = 20
debate_prompt_version = 1
judge_prompt_version = 1
batch_size=8
filter_size=4
for model in model_name:
    output_root = f"output/DEBATE/{model}"
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    for mask in MASK:
        data_path = "/mnt/sdc1/junhong/proj/w2s/dataset/sampled_saferlhf/alpaca_debate_10000.json"

        for round in ROUND:
            model_path = os.path.join(model_root, model)
            output_path = os.path.join(output_root, f"round{round}_{'mask' if mask else 'unmask'}.json")
            # print(output_path)
            cmd = f"python debate.py --filter_size {filter_size} --judge_prompt_version {judge_prompt_version} --model_path {model_path} --data_path {data_path} --output_path {output_path} --batch_size {batch_size} --round {round} --debate_prompt_version {debate_prompt_version}"
            if mask:
                cmd += " --mask"
            print("-" * 50)
            print(cmd)
            print("-" * 50)
            os.system(cmd)
            # print()
            data_path = output_path
