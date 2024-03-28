import os
import platform
import torch
from transformers import AutoTokenizer, AutoModel
import sys
from pathlib import Path

# MODEL_PATH = os.environ.get('MODEL_PATH', '/root/models/chatglm3-6b')
# MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')
# TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)
# tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
# model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()


import sys
# 添加 finetune_demo 的父目录到 sys.path
sys.path.append('/root/ChatGLM3')

#从finetune_demo包中导入inference.ht的load_model_and_tokenizer方法
from finetune_demo.inference_hf import load_model_and_tokenizer

# 获取环境变量中的MODEL_PATH，如果没有，则使用微调后的模型
MODEL_PATH = os.environ.get('MODEL_PATH', '/root/ChatGLM3/finetune_demo/output/checkpoint-3000')

# 调用inference.py中微调后的模型
model, tokenizer = load_model_and_tokenizer(MODEL_PATH) 

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

welcome_prompt = "欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"


def build_prompt(history):
    prompt = welcome_prompt
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM3-6B：{response}"
    return prompt


def main():
    past_key_values, history = None, []
    global stop_stream
    print(welcome_prompt)
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            past_key_values, history = None, []
            os.system(clear_command)
            print(welcome_prompt)
            continue
        print("\nChatGLM：", end="")
        current_length = 0
        for response, history, past_key_values in model.stream_chat(tokenizer, query, history=history, top_p=1,
                                                                    temperature=0.01,
                                                                    past_key_values=past_key_values,
                                                                    return_past_key_values=True):
            if stop_stream:
                stop_stream = False
                break
            else:
                print(response[current_length:], end="", flush=True)
                current_length = len(response)
        print("")


if __name__ == "__main__":
    main()