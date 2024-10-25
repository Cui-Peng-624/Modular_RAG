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
# Query transformations 02: RAG-fusion - 暂时还有点疑问
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
import os
os.environ["http_proxy"]="127.0.0.1:7890"
os.environ["https_proxy"]="127.0.0.1:7890"

from dotenv import load_dotenv
load_dotenv() # 加载 .env 文件
ZETATECHS_API_KEY = os.getenv('ZETATECHS_API_KEY')
ZETATECHS_API_BASE = os.getenv('ZETATECHS_API_BASE')

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

# 这里暂时不检验生成的sub-questions的格式
def decompsition_generate_queries(question: str, num_to_generate: int = 3, model_name: str = "gpt-4o-mini", temperature: float = 0) ->  list:
    template = f"""You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Generate multiple search queries related to: {question} \n
    Output ({num_to_generate} queries):"""
    prompt_decomposition = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(api_key = ZETATECHS_API_KEY, base_url = ZETATECHS_API_BASE, model = model_name, temperature = temperature)
    generate_queries_decomposition = ( prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n"))) # chain
    generated_questions = generate_queries_decomposition.invoke({"question":question})
    return generated_questions

sub_question_with_retrieved_docs_template = """"
Question: {sub_question}
Retrieved Documents:
{documents}
        
Please provide an answer based on the retrieved information and question.
"""

sub_question_with_retrieved_docs_prompt_template = ChatPromptTemplate.from_template(sub_question_with_retrieved_docs_template)

def format_qa_pair(question: str, answer: str) -> str:
    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
    # return formatted_string.strip()
    return formatted_string

# 1. 定义生成最终Prompt的模板
template = """Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}
"""

decomposition_prompt = ChatPromptTemplate.from_template(template)

# 2. 将q_a_pairs和context转化为字符串并构造最终Prompt
def format_final_prompt(original_question: str, q_a_pairs: list, context: list) -> str:
    # 将 q_a_pairs 列表转换为字符串，每个pair前加上标识符
    q_a_pairs_str = "\n".join([f"Q&A Pair {i + 1}:\n{pair}" for i, pair in enumerate(q_a_pairs)])
    
    # 将 context 列表转换为字符串，每个文档前加上标识符
    context_str = "\n".join([f"Document {i + 1}:\n{doc}" for i, doc in enumerate(context)])
    
    # 生成最终的Prompt
    final_prompt = decomposition_prompt.format(question=original_question, q_a_pairs=q_a_pairs_str, context=context_str)
    return final_prompt

def decompsition_generate_final_prompt(original_question: str, generated_questions: list, docs_per_generated_question: int = 3, model_name: str = "gpt-4o-mini", temperature: float = 0) -> str:
    """
    每个循环sub-questions：
        1. 构建q_a_pairs - （1）到vb中匹配与sub-questions相似的前docs_per_generated_question个文档，（2）将sub-questions和匹配的文档组合成prompt，（3）将prompt输入大模型获得answer，（4）将sub-questions和answer组合成q_a_pairs
        2. 构建context - 这个简单，每次检索完extend到一个list，最后list(set())去重即可
    """
    pinecone_manager = PineconeManager()
    q_a_pairs = []
    context = []
    for generated_question in generated_questions:
        _, results_only_str = pinecone_manager.retrieval(generated_question, top_k=docs_per_generated_question) # 返回的都是list
        combined_docs = "\n".join([f"{i + 1}. {doc}" for i, doc in enumerate(results_only_str)])
        llm = ChatOpenAI(api_key = ZETATECHS_API_KEY, base_url = ZETATECHS_API_BASE, model = model_name, temperature = temperature)
        generate_transition_model_answer_chain = ( sub_question_with_retrieved_docs_prompt_template | llm | StrOutputParser() ) # chain
        generate_transition_model_answer = generate_transition_model_answer_chain.invoke({"sub_question":generated_question, "documents":combined_docs}) # 得到了generated_question对应的answer
        q_a_pair = format_qa_pair(generated_question, generate_transition_model_answer) # 构建了q_a_pairs
        q_a_pairs.append(q_a_pair)
        context.extend(results_only_str) # 构建了context

    context = list(set(context)) # 去重

    # 构建最终的Prompt
    final_prompt = format_final_prompt(original_question, q_a_pairs, context)

    return final_prompt

def decompsition_generate(final_prompt: str, model_name: str = "gpt-4o-mini", temperature: float = 0.3) -> str:
    template = """{final_prompt}"""
    prompt_decomposition = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(api_key = ZETATECHS_API_KEY, base_url = ZETATECHS_API_BASE, model = model_name, temperature = temperature)
    generate_answer = ( prompt_decomposition | llm | StrOutputParser() ) # chain
    generated_answer = generate_answer.invoke({"final_prompt":final_prompt})
    return generated_answer

##########################################################################################################################################
# Query transformations 04: Step Back
##########################################################################################################################################





