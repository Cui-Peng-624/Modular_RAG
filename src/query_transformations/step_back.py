# 添加项目根目录到sys.path
from pathlib import Path
import sys
project_root = str(Path(__file__).parent.parent.absolute()) # e:\RAG\src\query_transformations\multi_query.py\..\..
sys.path.append(project_root)

from model_utils.model_manage import ModelManage # type: ignore

def validate_and_clean_output(output: str) -> str:
    """
    验证并清理模型输出，确保只返回干净的问题文本。

    Args:
        output (str): 模型输出的文本

    Returns:
        str: 清理后的文本
    """
    # 移除可能的前缀（如 "Step-Back Question:" 等）
    undesired_prefixes = ["Step-Back Question:", "Q:", "Question:"]
    for prefix in undesired_prefixes:
        if output.startswith(prefix):
            output = output[len(prefix):].strip()

    # 去掉额外的空格
    return output.strip()

def step_back_question(
    user_question: str,
    model_type: str = "api",
    model_name: str = "gpt-4o-mini",
) -> str:
    """
    将用户的具体问题转换为更通用、更抽象的Step-Back问题。

    Args:
        user_question (str): 用户的原始问题
        model_type (str): 模型类型 (默认使用 "api")
        model_name (str): 模型名称 (默认使用 "gpt-4o-mini")

    Returns:
        str: 生成的Step-Back问题
    """

    # 构建 prompt（自定义提示词）
    final_user_prompt = f"""
    You are an expert at world knowledge. Your task is to step back and paraphrase a user question, turning it into a more generic step-back question that is easier to answer. 

    Here are some examples of step-back questions:
    - Input: Could the members of The Police perform lawful arrests?
      Step-Back Question: what can the members of The Police do?
    - Input: Jan Sindel's was born in what country?
      Step-Back Question: what is Jan Sindel’s personal history?
    - Input: How does climate change affect polar bears?
      Step-Back Question: What are the effects of climate change on species?

    Please follow this instruction:
    1. Rewrite the user's question into a Step-Back Question that is generic and easier to answer.
    2. Use clear and concise language.
    3. Ensure the question represents a more general perspective.
    4. Make sure the output is concise and directly outputs a question in natural language with no prefixes.
    
    Now, process the following question:
    Input Question: {user_question}
    Step-Back Question: 
    """

    # 调用自定义的模型管理工具
    model = ModelManage(model_type=model_type, model_name=model_name)
    response = model.generate(final_user_prompt, mode="normal", model_name=model_name)

    # 验证输出格式，清理多余内容
    step_back_question_clean = validate_and_clean_output(response)

    # 返回生成的 Step-Back 问题
    return step_back_question_clean