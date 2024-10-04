import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7,0"
import numpy as np

np.random.seed(42)
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, AutoModelForSequenceClassification, \
    Trainer
import datasets
import torch
# from trl import
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM,SFTConfig
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from peft import LoraConfig, PeftModel
from scipy.stats import entropy
from transformers.generation import GenerationConfig
import json


def load_model(base_model_path, peft_model_path=None):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto")
    # if os.path.exists(peft_model_path):
    #     model.load_adapter(peft_model_path)
    return model, tokenizer


def train_model(model, tokenizer, data_path, formatting_prompts_func):
    # ds_train = datasets.Dataset.from_pandas(df_train)
    # ds_val = datasets.Dataset.from_pandas(df_val)
    data = datasets.Dataset.from_json(data_path)
    split_data = data.train_test_split(test_size=0.2)
    ds_train, ds_val = split_data["train"], split_data["test"]
    training_args = TrainingArguments(
        output_dir='models/tmp',
        per_device_train_batch_size=1,
        gradient_accumulation_steps=24,
        learning_rate=args.lr,
        logging_steps=100,
        remove_unused_columns=True, )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # sft_config = SFTConfig(max_seq_length=1600)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        # max_seq_length=1600,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        max_seq_length=2048
        # sft_config=sft_config,
        # data_collator=data_collator,
    )

    trainer.train()

    return trainer


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        if example['better_response_id'] == 0:
            text = example['prompt'][i] + " " + example['response_0'][i] + tokenizer.eos_token
        else:
            text = example['prompt'][i] + " " + example['response_1'][i] + tokenizer.eos_token
        output_texts.append(text)
    # print("examples of the SFT data")
    # print(output_texts[:10])
    return output_texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/mnt/sdc1/junhong/proj/w2s/output/DEBATE/debug.json",
                        help="path to the data file.", )
    parser.add_argument("--model_path", type=str, default="/mnt/sdc1/ModelWarehouse/Meta-Llama-3-8B-Instruct",
                        choices=["Mistral-7B", "LLAMA2-13B", "LLAMA3-8B"])
    parser.add_argument("--peft_model_path", type=str, default="output/sft_model/debug",
                        help="path to the peft model file.", )
    parser.add_argument("--lr", default=0.00005, type=float, help="learning rate for training the model.", )
    args = parser.parse_args()
    with open(args.data_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    model, tokenizer = load_model(args.model_path)
    # train model
    # response_template = ". ### Answer:"
    # response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
    # collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    sft_trainer = train_model(model, tokenizer, args.data_path, formatting_prompts_func)
    sft_trainer.save_model(args.peft_model_path)
