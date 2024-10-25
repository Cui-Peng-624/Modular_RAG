from ModelManage import ModelManage
from PineconeManager import PineconeManager

##########################################################################################################################################
# Query transformations 01: Multi query
##########################################################################################################################################
def multi_query_validate_format(questions: list) -> bool:
    """ 验证生成的结果是否符合格式要求 - 每个问题都以数字和句点开头 """
    for idx, question in enumerate(questions):
        # 去掉前后空格，并确保编号后有空格
        expected_start = f"{idx + 1}."
        if not question.strip().startswith(expected_start):
            return False
    return True

def multi_query_generate_queries(user_prompt: str, num_to_generate: str = "five", model_type: str = "api", model_name: str = "gpt-4o-mini") -> list:
    # prompt from langchain course
    # final_user_prompt = f"You are an AI language model assistant. Your task is to generate {num_to_generate} different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines. Original question: {user_prompt}" 
    # prompt from gpt upgraded
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

    model = ModelManage(model_type = model_type, model_name = model_name)
    response = model.generate(final_user_prompt)

    # 将生成的响应按行分割，并去掉空白行
    result_with_numbers = [item.strip() for item in response.split("\n") if item.strip()]
    
    # 验证格式（包括编号）
    if not multi_query_validate_format(result_with_numbers):
        # raise ValueError("Generated questions do not meet the required format.")
        print("Generated questions do not meet the required format!")
        return result_with_numbers
    
    # 移除编号和点，仅保留问题文本
    result = [question.split('. ', 1)[1] for question in result_with_numbers]

    return result

# 利用multi_query()函数生成的多个问题在数据库中进行检索，并返回唯一的结果
def multi_query_retrieve(generated_questions: list, top_k: int = 3) -> list:
    pinecone_manager = PineconeManager()
    results = []
    for generated_question in generated_questions:
        results_with_metadata, results_only_str = pinecone_manager.retrieval(generated_question, top_k=top_k)
        results.extend(results_only_str)
    return list(set(results)) # 返回唯一的结果

# 利用multi_query_retrieve()函数检索到的结果，结合原始问题
def multi_query_generate(original_user_question: str, info_retrieved: list, model_type: str = "api", model_name = "gpt-4o-mini") -> str:
    """
    将原始问题和从数据库中检索到的信息整合，并发送给大模型生成最终的回答。
    
    参数:
    - original_user_question: 用户的原始问题。
    - info_retrieved: 从向量数据库中检索到的相关信息列表。
    - model_type: 使用的模型类型，默认为 "api"。
    - model_name: 使用的模型名称，默认为 "gpt-4o-mini"。

    返回:
    - 大模型生成的回答。
    """
    
    # 整合原始问题和检索到的信息
    combined_prompt = f"User's question: {original_user_question}\n\n"
    combined_prompt += "Here is the relevant information retrieved from the database:\n"
    
    # 添加检索到的信息
    for idx, info in enumerate(info_retrieved):
        combined_prompt += f"{idx + 1}. {info}\n"
    
    combined_prompt += "\nUsing the user's question and the relevant information provided, please generate a comprehensive and informative answer."
    
    # 调用大模型生成答案
    model = ModelManage(model_type=model_type, model_name=model_name)
    model_response = model.generate(combined_prompt)
    
    return model_response

##########################################################################################################################################
# Query transformations 02: RAG-fusion
##########################################################################################################################################
from langchain.load import dumps, loads

def rag_fusion_validate_format(questions: list) -> bool:
    """ 验证生成的查询格式是否符合要求 - 每个问题都以数字和句点开头 """
    for idx, question in enumerate(questions):
        expected_start = f"{idx + 1}."
        if not question.strip().startswith(expected_start):
            return False
    return True

def rag_fusion_generate_queries(user_prompt: str, num_to_generate: str = "five", model_type: str = "api", model_name: str = "gpt-4o-mini") -> list:
    final_user_prompt = f"""
    You are a helpful assistant that generates multiple search queries based on a single input query. 
    Generate {num_to_generate} related search queries for the following question:

    Original question: {user_prompt}
    """

    model = ModelManage(model_type=model_type, model_name=model_name)
    response = model.generate(final_user_prompt)

    # 将生成的响应按行分割，并去掉空白行
    result_with_numbers = [item.strip() for item in response.split("\n") if item.strip()]

    # 验证格式
    if not rag_fusion_validate_format(result_with_numbers):
        print("Generated queries do not meet the required format!")
        return result_with_numbers

    # 移除编号和点，仅保留查询文本
    result = [question.split('. ', 1)[1] for question in result_with_numbers]
    
    return result

def rag_fusion_retrieve(generated_queries: list, top_k: int = 3) -> list:
    pinecone_manager = PineconeManager()
    results = []
    for query in generated_queries:
        results_with_metadata, results_only_str = pinecone_manager.retrieval(query, top_k=top_k)
        results.append(results_only_str)
    return results

def reciprocal_rank_fusion(results: list[list], k=60) -> list:
    """ 利用 RRF 重新排序多个查询返回的文档 """
    fused_scores = {}
    
    for docs_list in results:
        for rank, doc in enumerate(docs_list):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score) for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    return reranked_results

def rag_fusion_generate(original_user_question: str, info_retrieved: list, model_type: str = "api", model_name = "gpt-4o-mini") -> str:
    # 利用 RRF 对结果进行重新排序
    reranked_docs = reciprocal_rank_fusion(info_retrieved)

    # 整合结果
    combined_prompt = f"User's question: {original_user_question}\n\n"
    combined_prompt += "Here are the relevant information retrieved from the database:\n"

    for idx, (doc, _) in enumerate(reranked_docs):
        combined_prompt += f"{idx + 1}. {doc}\n"

    combined_prompt += "\nUsing the user's question and the relevant information provided, please generate a comprehensive and informative answer."

    # 调用大模型生成答案
    model = ModelManage(model_type=model_type, model_name=model_name)
    model_response = model.generate(combined_prompt)
    
    return model_response

##########################################################################################################################################
# Query transformations 03: Decomposition
##########################################################################################################################################