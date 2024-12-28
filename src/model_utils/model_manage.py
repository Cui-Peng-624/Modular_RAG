import os
os.environ["http_proxy"]="127.0.0.1:7890"
os.environ["https_proxy"]="127.0.0.1:7890"

# config.py
from dotenv import load_dotenv # type: ignore

# 加载 .env 文件
load_dotenv()

import asyncio
from typing import List, Dict, Any
# from langchain.llms import OpenAI, HuggingFacePipeline
# from langchain.chat_models import ChatOpenAI 
from openai import OpenAI # type: ignore
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline # type: ignore
# import torch # type: ignore



class ModelManage:
    def __init__(self, model_type: str = "api", model_name: str = "gpt-4o-mini"): # 默认使用api模型
        self.ZETATECHS_API_KEY = os.getenv('ZETATECHS_API_KEY')
        self.ZETATECHS_API_BASE = os.getenv('ZETATECHS_API_BASE')
        self.model_type = model_type # api or local
        self.model_name = model_name # gpt-4o-mini(api) or gemma-2-9b(local)
        self.model = self._initialize_model()

    def _initialize_model(self):
        if self.model_type == "api":
            client = OpenAI(api_key=self.ZETATECHS_API_KEY, base_url=self.ZETATECHS_API_BASE)
            return client
        # elif self.model_type == "local":
        #     if self.model_name == "gemma-2-9b": # https://huggingface.co/google/gemma-2-9b
        #         # running with the pipeline API
        #         pipe = pipeline(
        #             "text-generation",
        #             model="google/gemma-2-9b",
        #             device="cuda",  # replace with "mps" to run on a Mac device
        #         )
        #         return pipe
                
        #         # Running the model on a single / multi GPU
        #         # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
        #         # model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b", device_map="auto")
        #         # return tokenizer, model
        #     else:
        #         raise ValueError(f"Unsupported model type: {self.model_type}. We only support 'gemma-2-9b' in this task.")

    # 支持正常模式和结构化输出（json mode）模式
    def generate(self, user_prompt: str, mode: str = "normal", model_name: str = "gpt-4o-mini-2024-07-18", **kwargs) -> str:
        """
        该generate函数用于生成任意回答，仅需要输入user_prompt即可。
        
        mode: 
            normal: 正常模式
            structured: 结构化输出模式
        
        kwargs包含但不限于：system_prompt，max_tokens
        """
        # 获取特定的关键字参数
        system_prompt = kwargs.get("system_prompt", "You are a helpful assistant.")  # 如果没有提供 system_prompt 参数，则使用默认值
        max_tokens = kwargs.get("max_tokens", 512)  # 如果没有提供 max_tokens 参数，则使用默认值 256
        response_format = kwargs.get("response_format", None) # 获取json mode所必须的response_format参数

        messages_to_model = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        if mode == "normal": # 如果是normal模式，则不需要使用传入的model_name参数
            if self.model_type == "api":
                response = self.model.chat.completions.create(model=model_name, messages=messages_to_model)
                return response.choices[0].message.content
            elif self.model_type == "local":
                pass
        elif mode == "structured": # 如果是structured模式，则需要使用传入的model_name参数 - 本地大模型暂未支持json mode
            response = self.model.chat.completions.create(model=model_name, messages=messages_to_model, response_format=response_format)
            return response.choices[0].message.content

    def generate_with_context(self, user_query: str, context_chunks: list, model_name: str = "gpt-4o-mini" , **kwargs) -> str:
        """
        结合用户查询和上下文chunks生成回答

        参数:
            user_query: str - 用户的查询/问题
            context_chunks: list - 从数据库检索到的相关文本块列表
            **kwargs: 传递给 generate 函数的额外参数

        返回:
            str: 生成的回答
        """
        # 将context chunks格式化组合，添加编号和分隔符
        formatted_chunks = []
        for i, chunk in enumerate(context_chunks):
            # 移除多余的空白字符并格式化文本块
            cleaned_chunk = " ".join(chunk.split())
            formatted_chunk = f"[文档片段 {i}]\n{cleaned_chunk}\n"
            formatted_chunks.append(formatted_chunk)
        
        # 使用分隔线连接chunks
        combined_context = "\n---\n".join(formatted_chunks)
        
        # 构建带有上下文的提示语
        prompt_template = f"""请基于以下背景信息回答用户的问题。背景信息按相关性排序，包含多个文档片段。

背景信息：
{combined_context}

用户问题：
{user_query}

请根据以上背景信息回答用户问题。如果引用特定文档片段的内容，请注明片段编号。如果背景信息中没有相关内容，请如实告知。请确保回答准确、完整且易于理解。"""

        # 添加默认的system prompt（如果在kwargs中没有指定）
        if "system_prompt" not in kwargs:
            kwargs["system_prompt"] = "你是一个乐于助人的AI助手。请基于提供的背景信息回答问题，保持准确性和客观性。引用信息时请标明来源。"

        # 调用generate函数生成回答
        response = self.generate(
            user_prompt=prompt_template,
            mode="normal",
            model_name=model_name,
            **kwargs
        )

        return response
    

        


