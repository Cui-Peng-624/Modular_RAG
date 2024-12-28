from pydantic import BaseModel # type: ignore

# 添加项目根目录到sys.path
from pathlib import Path
import sys
project_root = str(Path(__file__).parent.parent.absolute()) # e:\RAG\src\query_transformations\multi_query.py\..\..
sys.path.append(project_root)

from model_utils.model_manage import ModelManage # type: ignore
from vdb_managers.drop.chroma_manager_1 import ChromaManager

class SearchQueriesGenerator(BaseModel):
    user_request: str
    sub_questions: list[str]  # 改为 sub_questions

def generate_queries_decomposition_with_structured_output(
    user_prompt: str, 
    num_to_generate: int = 5, 
    model_type: str = "api", 
    model_name: str = "gpt-4o-mini-2024-07-18"
) -> list:
    """使用OpenAI的JSON模式输出将复杂问题分解为多个子问题
    
    Args:
        user_prompt: 用户原始问题
        num_to_generate: 生成子问题的数量
        model_type: 使用的模型类型
        model_name: 使用的模型名称（需要支持JSON模式输出）
        
    Returns:
        list: 生成的子问题列表
    """

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "queries_generator",
            "description": "Generates multiple sub-questions related to an input question to maximize relevant document retrieval from a vector database. The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. ", 
            "schema": { 
                "type": "object",
                "properties": {
                    "user_request": {
                        "type": "string",
                        "description": "The original complex question asked by the user.."
                    },
                    "sub_questions": {
                        "type": "array",
                        "description": f"""A list of {num_to_generate} sub-questions that:
                        1. Break down the original complex question into smaller, focused questions
                        2. Each sub-question should address a specific aspect of the original question
                        3. Can be answered independently in isolation
                        4. Together provide a comprehensive understanding of the original question
                        5. Are clear, specific and directly answerable""",
                        "items": {
                            "type": "string",
                            "description": "A focused sub-question that addresses one specific aspect of the original complex question."
                        },
                    }
                },
                "required": ["user_request", "sub_questions"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
    
    model = ModelManage(model_type=model_type, model_name=model_name)
    response = model.generate(user_prompt, mode="structured", response_format=response_format)

    try:
        temp = SearchQueriesGenerator.parse_raw(response).dict()
        return temp['sub_questions']
    except Exception as e:
        print(f"Error parsing response: {e}")
        return []
    
############################## Answer recursively ##############################
template_recursively = """Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}
"""

def generate_final_prompt_decomposition_recursively(user_prompt: str, model_type: str = "api", model_name: str = "gpt-4o-mini", mode: str = "hybrid", **kwargs) -> str:
    """
    mode：["hybrid", "dense"]
    kwargs: 其他参数，比如dense_weight
    """
    q_a_pairs = ""
    chroma = ChromaManager()
    model = ModelManage(model_type=model_type, model_name=model_name)
    sub_questions = generate_queries_decomposition_with_structured_output(user_prompt)
    for sub_question in sub_questions:

        sub_question_docs = chroma.search(sub_question, mode = mode, **kwargs) # 返回list[str]

        sub_question_answer = model.generate_with_context(sub_question, sub_question_docs)
        q_a_pairs += f"Question: {sub_question}\nAnswer: {sub_question_answer}\n\n"
    
    context = chroma.get_formatted_context(user_prompt, mode = mode)

    final_prompt = template_recursively.format(question=user_prompt, q_a_pairs=q_a_pairs, context=context)
    return final_prompt

############################## Answer individually ##############################
template_individually = """Here is a set of Q+A pairs:

{context}

Use these to synthesize an answer to the question: {question}
"""

def generate_final_prompt_decomposition_individually(user_prompt: str, model_type: str = "api", model_name: str = "gpt-4o-mini", mode: str = "hybrid", **kwargs) -> str:
    """
    将复杂问题分解为子问题，对每个子问题单独检索和回答，最后综合所有答案
    
    Args:
        user_prompt: 用户原始问题
        model_type: 使用的模型类型
        model_name: 使用的模型名称
        mode: 搜索模式，可选值为 ["hybrid", "similarity", "similarity_with_score"]
        **kwargs: 其他参数，比如dense_weight
        
    Returns:
        str: 最终的提示语，包含所有子问题的答案和原始上下文
    """
    # 初始化必要的组件
    chroma = ChromaManager()
    model = ModelManage(model_type=model_type, model_name=model_name)
    
    # 生成子问题
    sub_questions = generate_queries_decomposition_with_structured_output(user_prompt)
    
    # 存储所有子问题的答案
    answers = ""
    
    # 对每个子问题单独处理
    for sub_question in sub_questions:
        # 为子问题检索相关文档
        context = chroma.get_formatted_context(sub_question, mode=mode, **kwargs) # str
        
        # 构建子问题的提示模板
        sub_prompt = f"""请基于以下背景信息回答问题。

问题：{sub_question}

背景信息：
{context}

请基于上述背景信息回答问题。如果背景信息中没有相关内容，请如实说明。
"""
        # 生成子问题的答案
        sub_answer = model.generate(user_prompt=sub_prompt)
        # 保存问答对
        answers += f"Question: {sub_question}\nAnswer: {sub_answer}\n\n"
    
    final_prompt = template_individually.format(question=user_prompt, context=answers)
    return final_prompt



