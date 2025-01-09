import asyncio
from typing import List, Dict, Any
import nest_asyncio  # type: ignore  # 添加这一行

# 在 Jupyter Notebook 环境中启用嵌套事件循环
nest_asyncio.apply()

from .AsyncAPIClient import AsyncAPIClient

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