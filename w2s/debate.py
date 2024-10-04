import json
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '6,7'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from utils.qwen_generation_utils import make_context
from template import *
import argparse
from tqdm import tqdm
from template import *
from tqdm import trange


class Debate:
    def __init__(self, args):
        self.args = args
        self.data = self.get_data()
        self.model, self.tokenizer = self.init_model()
        self.mode = args.mode
        self.output_path = args.output_path
        self.round = args.round
        self.debate_prompt_version = args.debate_prompt_version - 1
        self.judge_prompt_version=args.judge_prompt_version - 1
        self.mask = args.mask
        self.template = self.get_template()
        # self.filter = args.filter
        self.filter_size=args.filter_size

    def init_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, padding_side='left', trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(self.args.model_path, device_map="auto",
                                                     trust_remote_code=True).eval()
        model.generation_config = GenerationConfig.from_pretrained(self.args.model_path,
                                                                   pad_token_id=tokenizer.pad_token_id)
        model.generation_config.do_sample = True
        # model.generation_config.max_new_tokens =
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        if tokenizer.eos_token is None:
            tokenizer.eos_token = '<|endoftext|>'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def get_data(self):
        with open(self.args.data_path, 'r') as f:
            self.result = []
            data = json.load(f)
            if self.args.length:
                # data = data[:self.args.length]
                data = data[:self.args.length]
        if os.path.exists(self.args.output_path):
            with open(args.output_path, 'r', encoding="utf-8") as f:
                self.result = json.load(f)
            if len(data) == len(self.result) or len(self.result) > len(data):
                exit()
            data = data[len(self.result):]

        return data

    def get_template(self):
        if self.round == 1:
            template = TEMPLATE["DEBATE"]["1"][self.debate_prompt_version]
        elif self.round == 2:
            template = TEMPLATE["DEBATE"]["2"][self.debate_prompt_version]
        elif self.round == 3:
            template = TEMPLATE["DEBATE"]["3"][self.debate_prompt_version]
        elif self.round == 4:
            template = TEMPLATE["DEBATE"]["4"][self.judge_prompt_version]
        else:
            raise ValueError("round should be in [1,2,3,4]")
        return template

    def get_prompt(self, data):
        def add_system(text, model_base="qwen"):
            if model_base == "qwen":
                system_prompt = f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n'
            else:
                raise NotImplementedError("other model not include yet.")
            return system_prompt

        PROMPT_0 = []
        PROMPT_1 = []
        PROMPT_2 = []
        for i in data:
            if self.mode == "preference":
                if self.round == 1:
                    prompt = add_system(self.template.format(i["prompt"], i["response_0"], i["response_1"]))
                    PROMPT_0.append(prompt)
                    PROMPT_1.append(prompt)
                elif self.round == 2 or self.round == 3:
                    if self.mask:
                        prompt = add_system(
                            self.template.format(i["prompt"], i["response_0"], i["response_1"],
                                                 i["argument_context_mask"]))
                        PROMPT_0.append(prompt)
                        PROMPT_1.append(prompt)
                    else:
                        prompt_0 = add_system(
                            self.template.format(i["prompt"], i["response_0"], i["response_1"],
                                                 i["argument_context_0"]))
                        prompt_1 = add_system(
                            self.template.format(i["prompt"], i["response_0"], i["response_1"],
                                                 i["argument_context_1"]))
                        PROMPT_0.append(prompt_0)
                        PROMPT_1.append(prompt_1)

                elif self.round == 4:
                    prompt_0 = add_system(
                        self.template.format(i["prompt"], i["response_0"], i["response_1"],
                                             i["argument_context_mask"].split("<Round2>")[0]))
                    prompt_1 = add_system(
                        self.template.format(i["prompt"], i["response_0"], i["response_1"],
                                             i["argument_context_mask"].split("<Round3>")[0]))
                    prompt_2 = add_system(
                        self.template.format(i["prompt"], i["response_0"], i["response_1"], i["argument_context_mask"]))
                    PROMPT_0.append(prompt_0)
                    PROMPT_1.append(prompt_1)
                    PROMPT_2.append(prompt_2)

            elif self.mode == 'normal':
                if self.round == 1:
                    prompt_0 = add_system(
                        self.template.format("Alice", "Bob", i["prompt"], 'answer_1', 'answer_2', i["response_0"],
                                             i["response_1"]))
                    prompt_1 = add_system(
                        self.template.format("Bob", "Alice", i["prompt"], 'answer_2', 'answer_1', i["response_1"],
                                             i["response_0"]))
                    PROMPT_0.append(prompt_0)
                    PROMPT_1.append(prompt_1)

                elif self.round == 2 or self.round == 3:
                    if not self.mask:
                        prompt_0 = add_system(
                            self.template.format("Alice", "Bob", i["prompt"], i["response_0"], i["response_1"],
                                                 'answer_1', 'answer_2', i["argument_context_0"]))
                        prompt_1 = add_system(
                            self.template.format("Bob", "Alice", i["prompt"], i["response_1"], i["response_0"],
                                                 'answer_2', 'answer_1', i["argument_context_1"]))
                        PROMPT_0.append(prompt_0)
                        PROMPT_1.append(prompt_1)
                    else:
                        raise Exception("Mask not supported in normal mode")

                elif self.round == 4:
                    prompt_0 = add_system(
                        self.template.format(i["prompt"], i["response_0"], i["response_1"],
                                             i["argument_context_mask"].split("<Round2>")[0]))
                    prompt_1 = add_system(
                        self.template.format(i["prompt"], i["response_0"], i["response_1"],
                                             i["argument_context_mask"].split("<Round3>")[0]))
                    prompt_2 = add_system(
                        self.template.format(i["prompt"], i["response_0"], i["response_1"], i["argument_context_mask"]))
                    PROMPT_0.append(prompt_0)
                    PROMPT_1.append(prompt_1)
                    PROMPT_2.append(prompt_2)

        if self.round == 4:
            return PROMPT_0, PROMPT_1, PROMPT_2
        else:
            return PROMPT_0, PROMPT_1

    def get_argument(self, data):
        argument = ["Argument:", "Argument*", "argument:", "argument*"]
        for a in argument:
            if a in data:
                processed_data = data.split(a)[-1].strip()
                return processed_data
        return ""

    def make_context(self, data):

        context_0 = ""
        context_1 = ""
        context_mask = ""
        if self.mode == 'preference':
            for idx, (i, j) in enumerate(zip(data["argument_0"], data["argument_1"])):
                context_0 += f"<Round{idx + 1}>\nYour_Argument:{i}Other_Argument:{j}<\Round{idx + 1}>\n"
                context_1 += f"<Round{idx + 1}>\nYour_Argument:{j}Other_Argument:{i}<\Round{idx + 1}>\n"
                context_mask += f"<Round{idx + 1}>\nArgument_1:{j}Argument_2:{i}<\Round{idx + 1}>\n"
        elif self.mode == 'normal':
            for idx, (i, j) in enumerate(zip(data["argument_0"], data["argument_1"])):
                context_0 += f"<Round{idx + 1}>\nAlice:{i}Bob:{j}<\Round{idx + 1}>\n"
                context_1 += f"<Round{idx + 1}>\nBob:{j}Alice:{i}<\Round{idx + 1}>\n"
                context_mask += f"<Round{idx + 1}>\nArgument_1:{j}Argument_2:{i}<\Round{idx + 1}>\n"
        else:
            raise NotImplementedError("mode should be preference or normal")

        return {"context_0": context_0, "context_1": context_1, "context_mask": context_mask}

    def calculate_entropy(self, data):
        pass
        # return calculate_entropy(data)
    def get_entropy(self, data):
        return self.calculate_entropy(data)
    def inference_batch(self, batch_size=4):
        def inference(processed_prompt):
            inputs = self.tokenizer(processed_prompt, return_tensors='pt', padding=True)
            inputs = inputs.to(self.model.device)
            outputs = self.model.generate(**inputs)
            outputs = self.tokenizer.batch_decode(outputs[:, inputs['input_ids'].size(1):], skip_special_tokens=True)
            return outputs

        if self.round <= 3:
            for i in trange(0, len(self.data), batch_size):
                prompt_0, prompt_1 = self.get_prompt(self.data[i:i + batch_size])
                outputs_0 = inference(prompt_0)
                outputs_1 = inference(prompt_1)

                for j, (output_0, output_1) in enumerate(zip(outputs_0, outputs_1)):
                    if self.round == 1:
                        self.data[i + j]["model_response_0"] = [output_0.strip()]
                        self.data[i + j]['model_response_1'] = [output_1.strip()]
                        self.data[i + j]['argument_0'] = [self.get_argument(self.data[i + j]['model_response_0'][-1])]
                        self.data[i + j]['argument_1'] = [self.get_argument(self.data[i + j]['model_response_1'][-1])]

                    else:
                        self.data[i + j]["model_response_0"].append(output_0.strip())
                        self.data[i + j]["model_response_1"].append(output_1.strip())
                        self.data[i + j]['argument_0'].append(
                            self.get_argument(self.data[i + j]['model_response_0'][-1]))
                        self.data[i + j]['argument_1'].append(
                            self.get_argument(self.data[i + j]['model_response_1'][-1]))

                    context = self.make_context(self.data[i + j])
                    self.data[i + j]['argument_context_0'] = context["context_0"]
                    self.data[i + j]['argument_context_1'] = context["context_1"]
                    self.data[i + j]['argument_context_mask'] = context["context_mask"]
                    # self.data[i + j]["generate_prompt"]=[prompt_0[j],prompt_1[j]]
                    self.result.append(self.data[i + j])
                with open(self.args.output_path, 'w', encoding='utf-8') as f:
                    json.dump(self.result, f, ensure_ascii=False, indent=4)

        elif self.round == 4:

            for i in trange(0, len(self.data), batch_size):
                for m in range(self.filter_size):
                    prompt_0, prompt_1, prompt_2 = self.get_prompt(self.data[i:i + batch_size])
                    outputs_0 = inference(prompt_0)
                    outputs_1 = inference(prompt_1)
                    outputs_2 = inference(prompt_2)
                    for j, (output_0, output_1, output_2) in enumerate(zip(outputs_0, outputs_1, outputs_2)):
                        # self.data[i + j]["judge_model_response"] = []
                        # self.data[i + j]["judge_model_response"].append(output_0.strip())
                        # self.data[i + j]["judge_model_response"].append(output_1.strip())
                        # self.data[i + j]["judge_model_response"].append(output_2.strip())
                        # self.result.append(self.data[i + j])
                        if "judge_model_response_round1" not in self.data[i + j].keys():
                            self.data[i + j]["judge_model_response_round1"] = [output_0.strip()]
                            self.data[i + j]["judge_model_response_round2"] = [output_1.strip()]
                            self.data[i + j]["judge_model_response_round3"] = [output_2.strip()]
                        else:
                            self.data[i + j]["judge_model_response_round1"].append(output_0.strip())
                            self.data[i + j]["judge_model_response_round2"].append(output_1.strip())
                            self.data[i + j]["judge_model_response_round3"].append(output_2.strip())
                for j in range(batch_size):
                    self.result.append(self.data[i + j])
                with open(self.args.output_path, 'w', encoding='utf-8') as f:
                    json.dump(self.result, f, ensure_ascii=False, indent=4)


    def inference_filter_batch(self, batch_size=4):
        def inference(processed_prompt):
            inputs = self.tokenizer(processed_prompt, return_tensors='pt', padding=True)
            inputs = inputs.to(self.model.device)
            outputs = self.model.generate(**inputs)
            outputs = self.tokenizer.batch_decode(outputs[:, inputs['input_ids'].size(1):], skip_special_tokens=True)
            return outputs

        if self.round == 4:
            for i in trange(0, len(self.data), batch_size):
                prompt_0, prompt_1, prompt_2 = self.get_prompt(self.data[i:i + batch_size])
                outputs_0 = inference(prompt_0)
                outputs_1 = inference(prompt_1)
                outputs_2 = inference(prompt_2)
                for j, (output_0, output_1, output_2) in enumerate(zip(outputs_0, outputs_1, outputs_2)):
                    self.data[i + j]["judge_model_response"] = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='/mnt/sdc1/ModelWarehouse/qwen-7b-chat')
    parser.add_argument("--data_path", type=str,
                        default="/mnt/sdc1/junhong/proj/w2s/output/DEBATE/qwen-7b-chat/round3_mask.json")
    parser.add_argument("--length", type=int, help="采样的数据条数")
    parser.add_argument("--output_path", type=str, default="output/draft.json")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--mode", type=str, default="preference")
    parser.add_argument("--mask", action='store_true', default=False)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--debate_prompt_version", type=int, default=1)
    parser.add_argument("--judge_prompt_version", type=int, default=1)
    parser.add_argument("--filter_size", type=int, default=1)
    args = parser.parse_args()

    debater = Debate(args)
    debater.inference_batch(batch_size=args.batch_size)
