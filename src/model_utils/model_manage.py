import os
os.environ["http_proxy"]="127.0.0.1:7890"
os.environ["https_proxy"]="127.0.0.1:7890"

# config.py
from dotenv import load_dotenv # type: ignore

# 加载 .env 文件
load_dotenv()

# from typing import List, Dict, Any
# from langchain.llms import OpenAI, HuggingFacePipeline
# from langchain.chat_models import ChatOpenAI 
from openai import OpenAI # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline # type: ignore
import torch # type: ignore

class ModelManage:
    def __init__(self, model_type: str = "api", model_name: str = "gpt-4o-mini"): # 默认使用api模型
        self.ZETATECHS_API_KEY = os.getenv('ZETATECHS_API_KEY')
        self.ZETATECHS_API_BASE = os.getenv('ZETATECHS_API_BASE')
        # xprint(self.ZETATECHS_API_KEY, self.ZETATECHS_API_BASE)
        self.model_type = model_type # api or local
        self.model_name = model_name # gpt-4o-mini(api) or gemma-2-9b(local)
        self.model = self._initialize_model()

    def _initialize_model(self):
        if self.model_type == "api":
            client = OpenAI(api_key=self.ZETATECHS_API_KEY, base_url=self.ZETATECHS_API_BASE)
            return client
        elif self.model_type == "local":
            if self.model_name == "gemma-2-9b": # https://huggingface.co/google/gemma-2-9b
                # running with the pipeline API
                pipe = pipeline(
                    "text-generation",
                    model="google/gemma-2-9b",
                    device="cuda",  # replace with "mps" to run on a Mac device
                )
                return pipe
                
                # Running the model on a single / multi GPU
                # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
                # model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b", device_map="auto")
                # return tokenizer, model
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}. We only support 'gemma-2-9b' in this task.")

    def generate(self, user_prompt: str, **kwargs) -> str: # 该generate函数用于生成任意回答，仅需要输入user_prompt即可。kwargs包含但不限于：system_prompt，max_tokens
        # 获取特定的关键字参数
        system_prompt = kwargs.get("system_prompt", "You are a helpful assistant.")  # 如果没有提供 system_prompt 参数，则使用默认值
        max_tokens = kwargs.get("max_tokens", 256)  # 如果没有提供 max_tokens 参数，则使用默认值 256

        if self.model_type == "api":
            messages_to_model = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            response = self.model.chat.completions.create(model=self.model_name, messages=messages_to_model)
            return response.choices[0].message.content
        elif self.model_type == "local":
            outputs = self.model(user_prompt, max_new_tokens=max_tokens)
            response = outputs[0]["generated_text"]
            return response