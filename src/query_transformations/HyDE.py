import os
os.environ["http_proxy"]="127.0.0.1:7890"
os.environ["https_proxy"]="127.0.0.1:7890"

# config.py
from dotenv import load_dotenv # type: ignore

# 加载 .env 文件
load_dotenv()

ZETATECHS_API_KEY = os.getenv('ZETATECHS_API_KEY')
ZETATECHS_API_BASE = os.getenv('ZETATECHS_API_BASE')

from langchain.prompts import ChatPromptTemplate # type: ignore
from langchain_core.output_parsers import StrOutputParser # type: ignore
from langchain_openai import ChatOpenAI # type: ignore

# # 添加项目根目录到sys.path
# from pathlib import Path
# import sys
# project_root = str(Path(__file__).parent.parent.absolute()) # e:\RAG\src\query_transformations\multi_query.py\..\..
# sys.path.append(project_root)

# from model_utils.model_manage import ModelManage # type: ignore

def generate_hyde_document(
    user_question: str,
    model_name: str = "gpt-4o-mini",
) -> str:
    """
    基于用户输入的问题生成一个假设性文档，用于 HyDE（Hypothetical Document Embeddings）方法。

    Args:
        user_question (str): 用户输入的问题
        model_name (str): 使用的大语言模型名称，默认为 "gpt-4o-mini"

    Returns:
        str: 假设文档的内容，字符串格式
    """

    template = """Please write a scientific paper passage to answer the question
    Question: {question}
    Passage:"""
    prompt_hyde = ChatPromptTemplate.from_template(template)

    generate_docs_for_retrieval = (
        prompt_hyde | ChatOpenAI(api_key=ZETATECHS_API_KEY, base_url=ZETATECHS_API_BASE, model = model_name, temperature=0) | StrOutputParser()
    )
    
    generated_text = generate_docs_for_retrieval.invoke({"question":user_question})

    # 返回生成的假设文档内容
    return generated_text.strip()



