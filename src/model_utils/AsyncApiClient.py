import os
import json  # 用于将字典转换为 JSON 字符串
os.environ["http_proxy"] = "127.0.0.1:7897"
os.environ["https_proxy"] = "127.0.0.1:7897"

# config.py
from dotenv import load_dotenv  # type: ignore

# 加载 .env 文件
load_dotenv()

import asyncio
from typing import List, Dict, Any, Literal
from openai import AsyncOpenAI  # type: ignore
import nest_asyncio  # type: ignore  # 添加这一行

# 在 Jupyter Notebook 环境中启用嵌套事件循环
nest_asyncio.apply()

ModelType = Literal['chat', 'embedding']

class AsyncAPIClient:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.getenv('ZETATECHS_API_KEY'), 
            base_url=os.getenv('ZETATECHS_API_BASE')
        )
        self.semaphore = asyncio.Semaphore(10)  # 同时最多10个请求
        self.max_retries = 3
        self.retry_delay = 2

    async def generate_chat_response(self, prompt: str, response_format: Dict[str, Any]) -> str:
        """发送聊天请求到OpenAI API"""
        for attempt in range(1, self.max_retries + 1):
            try:
                async with self.semaphore:
                    response = await self.client.chat.completions.create(
                        model="gpt-4o-2024-08-06",
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        response_format=response_format
                    )
                    return response.choices[0].message.content
            except Exception as e:
                print(f"Chat请求失败 (尝试 {attempt}/{self.max_retries}): {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise Exception(f"Chat请求重试失败: {e}")

    async def generate_embedding(self, text: str) -> List[float]:
        """发送嵌入请求到OpenAI API"""
        for attempt in range(1, self.max_retries + 1):
            try:
                async with self.semaphore:
                    response = await self.client.embeddings.create(
                        model="text-embedding-3-large",
                        input=text
                    )
                    return response.data[0].embedding
            except Exception as e:
                print(f"Embedding请求失败 (尝试 {attempt}/{self.max_retries}): {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise Exception(f"Embedding请求重试失败: {e}")

    async def batch_generate(
        self, 
        inputs: List[str], 
        response_format: Dict[str, Any] = None, 
        model_type: ModelType = 'chat'
    ) -> List[Any]:
        """批量处理请求"""
        tasks = []
        for input_text in inputs:
            if model_type == 'chat':
                task = self.generate_chat_response(input_text, response_format)
            else:  # embedding
                task = self.generate_embedding(input_text)
            tasks.append(task)

        results = []
<<<<<<< HEAD
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                if model_type == 'chat':
                    # 将JSON字符串解析为字典
                    result = json.loads(result)
                results.append(result)
            except Exception as e:
                print(f"任务执行失败: {e}")
                results.append(None)

=======
        for prompt in prompts:
            result_str = await self.generate_response(prompt, response_format)
            # 将 JSON 字符串解析为字典
            result_dict = json.loads(result_str)
            results.append(result_dict)
>>>>>>> e503d4c6a721d0e7d96523baacc992414b9bc723
        return results