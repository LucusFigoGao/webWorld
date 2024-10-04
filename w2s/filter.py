import os
import json
import argparse
from scipy.stats import entropy


def get_label_round4(data):
    think = ["Thinking", "thinking", "Thought", "thought"]
    option_0 = ['Answer: 1', "Answer : 1"]
    option_1 = ['Answer: 2', 'Answer : 2']
    judge_model_response = []
    judge_model_response.extend(data["judge_model_response_round1"])
    judge_model_response.extend(data["judge_model_response_round2"])
    judge_model_response.extend(data["judge_model_response_round3"])
    label = {'0': 0, '1': 0}
    for i in judge_model_response:
        if any(opt in judge_model_response for opt in option_0):
            label['0'] += 1
        elif any(opt in judge_model_response for opt in option_1):
            label['1'] += 1
    return label


def get_label_round3(data):
    think = ["Thinking", "thinking", "Thought", "thought"]
    option_0 = ["Answer 1", "answer_1", "Answer 1", "Answer_1"]
    option_1 = ["Answer 2", "answer_2", "Answer 2", "Answer_2"]
    label = {'0': 0, '1': 0}
    debate_model_response = []
    debate_model_response.extend(data["model_response_0"][1:])
    debate_model_response.extend(data["model_response_1"][1:])
    for i in debate_model_response:
        for t in think:
            if t in i:
                answer = i.split(t)[0].strip()
                if any(opt in answer for opt in option_0):
                    label['0'] += 1
                    break
                elif any(opt in answer for opt in option_1):
                    label['1'] += 1
                    break
                else:
                    continue

    return label


def get_entropy(label):
    # 计算总数
    total_count = sum(label.values())

    # 将计数转换为概率
    probability_distribution = {key: count / total_count for key, count in label.items()}

    # 提取出概率分布的值
    probabilities = list(probability_distribution.values())
    if probabilities[0] >= probabilities[1]:
        sft_label = 0
    else:
        sft_label = 1
    # 使用 scipy.stats.entropy 计算信息熵
    information_entropy = entropy(probabilities)

    return information_entropy, sft_label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/mnt/sdc1/junhong/proj/w2s/output/DEBATE/qwen-7b-chat/round4_mask.json')
    parser.add_argument('--output_path', type=str, default='output/DEBATE/debug.json')
    args = parser.parse_args()

    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # print(len(data))

    trust_data = []
    for i in data:
        label = get_label_round3(i)
        if sum(label.values())==0:
            print(i)
            continue
        label_entropy, sft_label = get_entropy(label)
        # if entropy < 0.5:
        trust_data.append(
            {
                "prompt": i["prompt"],
                "response_0": i["response_0"],
                'response_1': i['response_1'],
                'is_response_0_safe': i['is_response_0_safe'],
                'is_response_1_safe': i['is_response_1_safe'],
                'better_response_id': i['better_response_id'],
                'safer_response_id': i['safer_response_id'],
                'entropy': label_entropy,
                'sft_label': sft_label
            }
        )
    print("Has entropy: ", len(trust_data))
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(trust_data, f, ensure_ascii=False, indent=4)
