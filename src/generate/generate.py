from pathlib import Path
import sys
project_root = str(Path(__file__).parent.parent.absolute()) # e:\RAG\src\sparse_retrievers\bm25_manager.py\..\..
# __file__ 是 Python 中的一个特殊变量，表示当前脚本的文件路径
sys.path.append(project_root)

from model_utils.model_manage import ModelManage

def generate_final_response(
    user_query: str, 
    retrieved_context: str,
    model_type: str = "api",
    model_name: str = "gpt-4o-mini-2024-07-18",
    **generation_args
) -> str:
    """
    RAG生成核心函数
    
    参数:
        user_query (str): 用户原始查询
        retrieved_context (str): 检索到的上下文文本
        model_type (str): 模型类型 api/local
        model_name (str): 模型名称
        **generation_args: 生成参数（temperature, max_tokens等）
        
    返回:
        str: 生成的最终回答
    """
    # 构建增强提示
    augmented_prompt = f"""基于以下上下文信息回答问题。如果上下文不相关，请使用常识进行回答。
    
【上下文】
{retrieved_context}

【问题】
{user_query}

请给出清晰、准确的回答，并引用上下文中的相关证据："""
    
    # 初始化模型管理器
    model_manager = ModelManage(model_type=model_type, model_name=model_name)
    
    # 调用生成接口
    return model_manager.generate(
        user_prompt=augmented_prompt,
        mode="normal",
        model_name=model_name,
        **{
            "system_prompt": "你是一个专业的问答助手，能够准确解析上下文并给出可靠回答",
            "max_tokens": 4096,
            **generation_args
        }
    )
