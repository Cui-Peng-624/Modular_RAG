import asyncio
from typing import List
import numpy as np # type: ignore
import nest_asyncio # type: ignore

nest_asyncio.apply()

from .AsyncAPIClient import AsyncAPIClient

async def extract_embeddings(texts: List[str]) -> List[np.ndarray]:
    """异步获取文本的嵌入向量"""
    # 初始化异步 API 客户端
    client = AsyncAPIClient()

    # 直接使用embedding模型
    results = await client.batch_generate(texts, model_type='embedding')
    
    # 转换结果为numpy数组
    embeddings = [np.array(result) for result in results if result is not None]
    
    return embeddings

def extract_embeddings_sync(texts: List[str]) -> List[np.ndarray]:
    """同步版本的 extract_embeddings"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(extract_embeddings(texts))