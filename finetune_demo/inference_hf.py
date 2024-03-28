#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Annotated, Union

import typer
import peft

from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

# 定义模型和分词器的类型别名，用于后续的类型提示
ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

# 创建 typer 应用实例
app = typer.Typer(pretty_exceptions_show_locals=False)

# 辅助函数，用于将路径字符串转换为 Path 对象，并解析为绝对路径
def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()

# 加载模型和分词器的函数
def load_model_and_tokenizer(model_dir: Union[str, Path]) -> tuple[ModelType, TokenizerType]:
    model_dir = _resolve_path(model_dir)
    # 检查模型目录下是否有特定的配置文件
    if (model_dir / 'adapter_config.json').exists():
        # 如果存在配置文件，说明是 PEFT 模型
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, device_map='auto'
        )
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        # 如果不存在配置文件，加载标准的因果语言模型
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, device_map='auto'
        )
        tokenizer_dir = model_dir
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=True
    )
    return model, tokenizer

# 定义 CLI 命令，该命令接受 model_dir 和 prompt 作为参数
@app.command()
def main(
        model_dir: Annotated[str, typer.Argument(help='')],
        prompt: Annotated[str, typer.Option(help='')],
):
   
    # 使用 load_model_and_tokenizer 函数加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(model_dir)
    # 使用模型生成对输入提示的响应
    response, _ = model.chat(tokenizer, prompt)
    # 打印响应内容
    print(response)

# 如果脚本作为主程序执行，启动 typer 应用
if __name__ == '__main__':
    app()
