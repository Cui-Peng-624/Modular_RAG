import os
import json  # 用于将字典转换为 JSON 字符串
os.environ["http_proxy"] = "127.0.0.1:7890"
os.environ["https_proxy"] = "127.0.0.1:7890"

# config.py
from dotenv import load_dotenv  # type: ignore

# 加载 .env 文件
load_dotenv()

import asyncio
from typing import List, Dict, Any
from openai import AsyncOpenAI  # type: ignore
import nest_asyncio  # type: ignore  # 添加这一行

# 在 Jupyter Notebook 环境中启用嵌套事件循环
nest_asyncio.apply()


class AsyncAPIClient:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv('ZETATECHS_API_KEY'), base_url=os.getenv('ZETATECHS_API_BASE'))  # AsyncOpenAI 专门用于异步
        # 限制并发请求数，避免触发API限制
        self.semaphore = asyncio.Semaphore(10)  # 同时最多10个请求
        self.max_retries = 3  # 设置最大重试次数
        self.retry_delay = 2  # 每次重试的延迟时间（秒）

    async def generate_response(self, prompt: str, response_format: Dict[str, Any]) -> str:
        """发送单个请求到OpenAI API，带重试机制，返回 JSON 字符串"""
        for attempt in range(1, self.max_retries + 1):
            try:
                # 使用信号量控制并发
                async with self.semaphore:  # 确保同时只有最多10个请求在处理
                    response = await self.client.chat.completions.create(
                        model="gpt-4o-mini-2024-07-18",  # 使用2024-07-18版本
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        response_format=response_format
                    )
                    return response.choices[0].message.content # str
            except Exception as e:
                print(f"请求失败 (尝试 {attempt}/{self.max_retries}): {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay)  # 等待一段时间后重试
                else:
                    # 如果达到最大重试次数，返回一个默认的 JSON 字符串
                    print(f"重试失败，返回默认值: {e}")
                    return {
                        "category": "未知",  # 默认类别
                        "keywords": []  # 默认关键字为空
                    }

    async def batch_generate(self, prompts: List[str], response_format: Dict[str, Any]) -> List[Dict[str, Any]]:
        """批量处理多个提示词，带重试机制，返回字典列表"""
        results = []
        for prompt in prompts:
            result_str = await self.generate_response(prompt, response_format)
            # 将 JSON 字符串解析为字典
            result_dict = json.loads(result_str)
            results.append(result_dict)
        return results