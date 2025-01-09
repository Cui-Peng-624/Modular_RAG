import os
import json
from typing import List, Dict, Any
import asyncio
from pathlib import Path
import sys

# 添加项目根目录到sys.path
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.append(project_root)

# 导入自定义的异步客户端
from model_utils.AsyncAPIClient import AsyncAPIClient

async def extract_summaries(documents: List[Any]) -> List[str]:
    """
    异步提取文档chunks的总结

    Args:
        documents (List[Any]): 
            - 文档的分块内容，每个chunk是一个对象，包含page_content属性
            - 形式类似：[Document(metadata={'source': 'files/xxx.pdf', 'page': 0}, page_content=)...]

    Returns:
        List[str]: 每个chunk的总结列表
    """
    # 初始化异步API客户端
    client = AsyncAPIClient()

    # 定义响应格式
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "extract_summary",
            "description": "提取文档内容的总结",
            "schema": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "文档内容的简要总结"
                    }
                },
                "required": ["summary"],
                "additionalProperties": False
            },
            "strict": True
        }
    }

    # 构建prompts
    prompts = []
    for chunk in documents:
        prompt = f"""
        请对以下文本内容进行简要总结，总结长度不要超过原文的1/3。
        请确保总结包含文本的主要观点和关键信息。

        文本内容：
        {chunk}
        """
        prompts.append(prompt)

    # 异步批量调用大模型API
    results = await client.batch_generate(prompts, response_format)
    
    # 提取所有总结
    summaries = [result["summary"] for result in results]
    
    return summaries

def extract_summaries_sync(documents: List[Any]) -> List[str]:
    """
    同步版本的extract_summaries，用于不支持异步的场景

    Args:
        documents (List[Any]): 文档分块列表

    Returns:
        List[str]: 每个chunk的总结列表
    """
    # 使用asyncio.get_event_loop().run_until_complete()运行异步函数
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(extract_summaries(documents))
