from typing import List
from langchain.docstore.document import Document # type: ignore

# 添加项目根目录到sys.path
from pathlib import Path
import sys
project_root = str(Path(__file__).parent.parent.absolute()) # e:\RAG\src\query_transformations\multi_query.py\..\..
sys.path.append(project_root)

from model_utils.model_manage import ModelManage # type: ignore
from vdb_managers.chroma_manager import ChromaManager

def validate_query_format_multi_query(questions: list) -> bool:
    """验证生成的结果是否符合格式要求 - 每个问题都以数字和句点开头
    
    Args:
        questions: 生成的问题列表
        
    Returns:
        bool: 是否符合格式要求
    """
    for idx, question in enumerate(questions):
        # 去掉前后空格，并确保编号后有空格
        expected_start = f"{idx + 1}."
        if not question.strip().startswith(expected_start):
            return False
    return True

def generate_queries_multi_query(user_prompt: str, num_to_generate: str = "five", model_type: str = "api", model_name: str = "gpt-4o-mini") -> list:
    """生成多个不同视角的查询
    
    Args:
        user_prompt: 用户原始问题
        num_to_generate: 生成问题的数量
        model_type: 使用的模型类型
        model_name: 使用的模型名称
        
    Returns:
        list: 生成的问题列表
    """
    # 构建 prompt
    final_user_prompt = f"""
    You are an AI language model assistant. Your task is to generate {num_to_generate} different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Each question should explore a different perspective or approach to the original question in order to maximize the diversity of search results.
    
    Here are the format and rules you must follow:
    1. Each question must begin with a number (e.g., 1., 2., etc.).
    2. Each question should vary in terms of phrasing, perspective, or specificity, while still maintaining relevance to the original question.
    3. Ensure the language is clear, concise, and grammatically correct.
    4. The goal is to generate alternative questions that may surface different relevant documents or results during vector-based search.

    Please provide the alternative questions separated by newlines.

    Original question: {user_prompt}
    """

    model = ModelManage(model_type=model_type, model_name=model_name)
    response = model.generate(final_user_prompt)

    # 将生成的响应按行分割，并去掉空白行
    result_with_numbers = [item.strip() for item in response.split("\n") if item.strip()]
    
    # 验证格式（包括编号）
    if not validate_query_format_multi_query(result_with_numbers):
        print("Generated questions do not meet the required format!")
        return result_with_numbers
    
    # 移除编号和点，仅保留问题文本
    result = [question.split('. ', 1)[1] for question in result_with_numbers]

    return result

# def retrieve_multi_query(generated_questions: list, collection_name: str = "default", top_k: int = 3) -> list:
#     """利用生成的多个问题检索文档
    
#     Args:
#         generated_questions: 生成的问题列表
#         collection_name: Chroma集合名称
#         top_k: 每个问题返回的结果数量
        
#     Returns:
#         list: 去重后的检索结果列表
#     """
#     chroma_manager = ChromaManager(collection_name=collection_name)
#     results = []
#     for question in generated_questions:
#         search_results = chroma_manager.similarity_search(question, k=top_k)
#         results.extend([result['content'] for result in search_results])
#     return list(set(results))  # 返回唯一的结果

# def generate_answer(original_question: str, retrieved_info: list, model_type: str = "api", model_name: str = "gpt-4") -> str:
#     """根据检索结果生成最终答案
    
#     Args:
#         original_question: 用户原始问题
#         retrieved_info: 检索到的相关信息列表
#         model_type: 使用的模型类型
#         model_name: 使用的模型名称
        
#     Returns:
#         str: 生成的回答
#     """
#     # 构建 prompt
#     combined_prompt = f"User's question: {original_question}\n\n"
#     combined_prompt += "Here is the relevant information retrieved from the database:\n"
    
#     # 添加检索到的信息
#     for idx, info in enumerate(retrieved_info):
#         combined_prompt += f"{idx + 1}. {info}\n"
    
#     combined_prompt += "\nUsing the user's question and the relevant information provided, please generate a comprehensive and informative answer."
    
#     # 调用大模型生成答案
#     model = ModelManage(model_type=model_type, model_name=model_name)
#     model_response = model.generate(combined_prompt)
    
#     return model_response

from pydantic import BaseModel # type: ignore

class SearchQueriesGenerator(BaseModel):
    user_request: str
    similar_questions: list[str]

def generate_queries_multi_query_with_structured_output(
    user_prompt: str, 
    num_to_generate: int = 5, 
    model_type: str = "api", 
    model_name: str = "gpt-4o-mini-2024-07-18"
) -> list:
    """使用OpenAI的JSON模式输出生成多个不同视角的查询
    
    Args:
        user_prompt: 用户原始问题
        num_to_generate: 生成问题的数量
        model_type: 使用的模型类型
        model_name: 使用的模型名称（需要支持JSON模式输出）
        
    Returns:
        list: 生成的问题列表
    """

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "queries_generator",
            "description": "Generate multiple search queries to maximize relevant document retrieval from a vector database", 
            "schema": { 
                "type": "object",
                "properties": {
                    "user_request": {
                        "type": "string",
                        "description": "The original question asked by the user."
                    },
                    "similar_questions": {
                        "type": "array",
                        "description": f"""A list of {num_to_generate} different versions of the user's question that:
                        1. Explore different perspectives or approaches to the original question
                        2. Help overcome limitations of distance-based similarity search
                        3. Vary in terms of phrasing, specificity, or focus
                        4. Maintain relevance to the original question
                        5. Use clear and concise language""",
                        "items": {
                            "type": "string",
                            "description": "A rephrased question that explores a different angle of the original question to maximize the diversity of search results."
                        },
                    }
                },
                "required": ["user_request", "similar_questions"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
    
    model = ModelManage(model_type=model_type, model_name=model_name)
    response = model.generate(user_prompt, mode="structured", response_format=response_format)

    try:
        temp = SearchQueriesGenerator.parse_raw(response).dict()
        return temp['similar_questions']
    except Exception as e:
        print(f"Error parsing response: {e}")
        return []

