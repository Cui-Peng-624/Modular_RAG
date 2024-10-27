import os
from dotenv import load_dotenv # type: ignore
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate # type: ignore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # type: ignore
from langchain_core.output_parsers import StrOutputParser # type: ignore
from langchain_chroma import Chroma # type: ignore
from langchain_core.example_selectors import SemanticSimilarityExampleSelector # type: ignore
from typing import List, Dict

# 设置代理
os.environ["http_proxy"] = "127.0.0.1:7890"
os.environ["https_proxy"] = "127.0.0.1:7890"

# 加载环境变量
load_dotenv()

class ExampleManager:
    def __init__(self, prompt_type: str = "fixed", vectordatabase: str = "chroma", k: int = 2):
        self.prompt_type = prompt_type
        self.vectordatabase = vectordatabase
        self.k = k
        self.examples = self.get_examples()
        self.api_key = os.getenv('ZETATECHS_API_KEY')
        self.base_url = os.getenv('ZETATECHS_API_BASE')
        self.embeddings = OpenAIEmbeddings(model = "text-embedding-3-large", api_key = self.api_key, base_url = self.base_url)
        self.vectorstore = None
        self.example_selector = None
        
        if self.prompt_type == "dynamic":
            self._setup_dynamic_selector(self.k)

    def get_examples(self) -> List[Dict[str, str]]:
        if self.prompt_type == "fixed":
            return self.static_examples()
        elif self.prompt_type == "dynamic":
            return self.static_examples()  # 初始使用静态示例，后续可动态添加
        else:
            raise ValueError("Invalid prompt_type. Choose 'fixed' or 'dynamic'.")

    def static_examples(self) -> List[Dict[str, str]]:
        return [
            {
                "input": "Could the members of The Police perform lawful arrests?",
                "output": "what can the members of The Police do?",
            },
            {
                "input": "Jan Sindel's was born in what country?",
                "output": "what is Jan Sindel's personal history?",
            },
        ]

    def _setup_dynamic_selector(self, k: int):
        to_vectorize = [" ".join(example.values()) for example in self.examples]
        self.vectorstore = Chroma.from_texts(to_vectorize, self.embeddings, metadatas=self.examples)
        self.example_selector = SemanticSimilarityExampleSelector(
            vectorstore=self.vectorstore,
            k=k,
        )

    def add_example(self, new_example: Dict[str, str]):
        self.examples.append(new_example)
        if self.prompt_type == "dynamic" and self.vectorstore is not None:
            to_vectorize = " ".join(new_example.values())
            self.vectorstore.add_texts([to_vectorize], metadatas=[new_example])

    def select_examples(self, input_text: str) -> List[Dict[str, str]]:
        if self.prompt_type == "fixed":
            return self.examples
        elif self.prompt_type == "dynamic":
            return self.example_selector.select_examples({"input": input_text})

class StepBackManager:
    def __init__(self, model_type: str = "api", model_name: str = "gpt-4o-mini", 
                 temperature: float = 0.3, prompt_type: str = "fixed"):
        self.api_key = os.getenv('ZETATECHS_API_KEY')
        self.base_url = os.getenv('ZETATECHS_API_BASE')
        self.llm = self._create_llm(model_type, model_name, temperature)
        self.example_manager = ExampleManager(prompt_type)
        self.prompt_template = self._create_prompt_template()

    def _create_llm(self, model_type: str, model_name: str, temperature: float):
        if model_type == "api":
            return ChatOpenAI(api_key=self.api_key, base_url=self.base_url, model=model_name, temperature=temperature)
        elif model_type == "local":
            # 实现本地模型的逻辑
            pass
        else:
            raise ValueError("Invalid model_type. Choose 'api' or 'local'.")

    def _create_prompt_template(self) -> ChatPromptTemplate:
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{output}"),
        ])

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_selector=self.example_manager.example_selector if self.example_manager.prompt_type == "dynamic" else None,
            example_prompt=example_prompt,
            examples=self.example_manager.examples if self.example_manager.prompt_type == "fixed" else [],
            input_variables=["input"] if self.example_manager.prompt_type == "dynamic" else [],
        )

        return ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
            ),
            few_shot_prompt,
            ("user", "{question}"),
        ])

    def generate_queries(self, question: str) -> List[str]:
        if self.example_manager.prompt_type == "dynamic":
            self.prompt_template.partial_variables["input"] = question

        generate_queries_step_back = (
            self.prompt_template 
            | self.llm 
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
        )
        return generate_queries_step_back.invoke({"question": question})

    def single_generate(self, question: str) -> str:
        # 实现生成单个step back question并回答的逻辑
        pass

    def multiple_generate(self, question: str) -> List[str]:
        # 实现多轮生成step back questions并逐个回答的逻辑
        pass

# 使用示例
if __name__ == "__main__":
    manager = StepBackManager(prompt_type="dynamic")
    
    # 添加新的示例
    manager.example_manager.add_example({
        "input": "What is the capital of France?",
        "output": "What are some important cities in France?"
    })

    question = "What was the impact of the Industrial Revolution on urban development?"
    results = manager.generate_queries(question)
    print(results)