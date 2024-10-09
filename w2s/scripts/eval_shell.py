import os

model_root = "/mnt/sdc1/ModelWarehouse"
model_name = ["Meta-Llama-3-8B-Instruct"]
peft_model_path = '/mnt/sdc1/junhong/proj/w2s/output/sft_model/debug'
batch_size=1
eval_prompt_version=1
data_path = "/mnt/sdc1/junhong/proj/w2s/dataset/sampled_saferlhf/alpaca_test_1000.json"
length=16
model_base = 'llama3'
for model in model_name:
    output_root = f"output/eval-DEBATE/{model}"
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    model_path = os.path.join(model_root, model)
    output_path = os.path.join(output_root, f"inference_result_{eval_prompt_version}.json")
    cmd = f"python eval.py " \
          f"--model_path {model_path} " \
          f"--peft_model_path {peft_model_path} " \
          f"--data_path {data_path} " \
          f"--output_path {output_path} " \
          f"--batch_size {batch_size} " \
          f"--eval_template_version {eval_prompt_version} " \
          f"--model_base {model_base} "
          # f"--length {length} " \


    print("-" * 50)
    print(cmd)
    print("-" * 50)
    os.system(cmd)

    # parser.add_argument("--model_path", type=str, default='/mnt/sdc1/ModelWarehouse/qwen-7b-chat')
    # parser.add_argument("--model_base",type=str, default='qwen')
    # parser.add_argument("--data_path", type=str,
    #                     default="/mnt/sdc1/junhong/proj/w2s/dataset/sampled_saferlhf/alpaca_test_1000.json")
    # parser.add_argument("--length", type=int, help="采样的数据条数")
    # parser.add_argument("--eval_template_version", type=int, default=2)
    # parser.add_argument("--output_path", type=str, default="output/draft.json")
    # parser.add_argument("--batch_size", type=int, default=4)


