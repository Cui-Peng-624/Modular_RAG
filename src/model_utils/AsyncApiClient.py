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
    
async def extract_metadata(file_path: str, documents_chunks: List[Any]) -> Dict[str, Any]:  
    """
    异步提取文档的类别和关键字

    Args:
        file_path (str): 文档路径
        documents_chunks (List[Any]): 
            - 文档的分块内容，每个 chunk 是一个对象，包含 page_content 属性
            - 形式类似： [Document(metadata={'source': 'files/论文 - GraphRAG.pdf', 'page': 0}, page_content=)
            - 可用 documents_chunks[i].page_content 获取 page_content

    Returns:
        Dict[str, Any]: 包含类别和关键字的字典，例如：
            {
                "category": "计算机",
                "keywords": ["机器学习", "自然语言处理", "深度学习"]
            }
    """
    # 初始化异步 API 客户端
    client = AsyncAPIClient()

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "extract_metadata",
            "description": "提取文档内容属于的大类以及其中的一些关键字",
            "schema": { 
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "文档属于的大类",
                        "enum": ["artifical_intelligence", "data_science", "other"] # 尽力列举所有的可能性。
                    },
                    "keywords":{
                        "type": "array",
                        "description": "文档中的一些关键字",
                        "items": {
                            "type": "string",
                            "description": "具体的一个关键字"
                        }
                    }
                },
                "required": ["category", "keywords"],
                "additionalProperties": False
            },
            "strict": True
        }
    }

    # 构建 prompts，提取类别和关键字
    prompts = []
    for chunk in documents_chunks:
        prompt = f"""
        请分析以下文本内容，并提取该文本所属的类别（如计算机、医学等）和关键字（不超过2个）。
        返回格式为JSON，包含两个字段：category（类别）和 keywords（关键字列表）。

        文本内容：
        {chunk.page_content}
        """
        prompts.append(prompt)

    # 异步批量调用大模型 API
    results = await client.batch_generate(prompts, response_format)

    return results

#################### 同步调用 ####################
def extract_metadata_sync(file_path: str, documents_chunks: List[Any]) -> Dict[str, Any]: # 解决了 Jupyter Notebook 中不支持异步的问题
    """
    同步版本的 extract_metadata，用于不支持异步的场景
    """
    # 使用 asyncio.run() 替换为 asyncio.get_event_loop().run_until_complete()
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(extract_metadata(file_path, documents_chunks))