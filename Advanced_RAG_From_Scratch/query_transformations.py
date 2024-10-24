from ModelManage import ModelManage

###########################################################################################
# Query transformations 01: Multi query
###########################################################################################
def multi_query_validate_format(questions: list) -> bool:
    """ 验证生成的结果是否符合格式要求 - 每个问题都以数字和句点开头 """
    for idx, question in enumerate(questions):
        # 去掉前后空格，并确保编号后有空格
        expected_start = f"{idx + 1}."
        if not question.strip().startswith(expected_start):
            return False
    return True

def multi_query(user_prompt: str, num_to_generate: str = "five") -> list:
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

    model = ModelManage(model_type="api", model_name="gpt-4o-mini")
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

# 现在需要整合multi_query
def multi_query_rag(generated_questions: list, rag_tokenizer, rag_model, num_results: int = 3) -> list:
    pass

###########################################################################################
# Query transformations 02: 
###########################################################################################
