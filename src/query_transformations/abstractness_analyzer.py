# 此组件专门用于判断用户输入的问题是否是抽象的，如果是抽象的，则需要进行抽象化处理
from pydantic import BaseModel # type: ignore

# 添加项目根目录到sys.path
from pathlib import Path
import sys
project_root = str(Path(__file__).parent.parent.absolute()) # e:\RAG\src\query_transformations\abstractness_analyzer.py\..\..
sys.path.append(project_root)

from model_utils.model_manage import ModelManage

class AbstractnessAnalyzer(BaseModel):
    abstract: bool # True表示抽象，False表示不抽象

response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "abstractness_analyzer",
        "description": """分析问题的抽象程度。以下是一些示例：

抽象问题示例：
- "人生的意义是什么？" (abstract=true，因为这是一个哲学性的抽象概念)
- "什么是幸福？" (abstract=true，因为幸福是一个主观且抽象的概念)
- "如何定义成功？" (abstract=true，因为成功的定义因人而异且抽象)

具体问题示例：
- "北京的人口是多少？" (abstract=false，因为这是可以用具体数字回答的问题)
- "怎么做红烧肉？" (abstract=false，因为这需要具体的步骤和材料)
- "苹果公司的市值是多少？" (abstract=false，因为这有具体的数值答案)

如果问题涉及抽象概念、哲学思考、主观判断，返回True；如果问题有具体答案、明确步骤或客观数据，返回False。""",
        "schema": { 
            "type": "object",
            "properties": {
                "abstract": {
                    "type": "boolean",
                    "description": "True表示问题更偏向抽象，False表示问题更偏向具体"
                },
            },
            "required": ["abstract"],
            "additionalProperties": False
        },
        "strict": True
    }
}

def abstractness_analyzer(question: str, model_name: str = "gpt-4o-mini-2024-07-18") -> bool:
    model = ModelManage()
    response = model.generate(question, mode="structured", model_name=model_name, response_format=response_format)
    try:
        temp = AbstractnessAnalyzer.parse_raw(response).dict()
        return temp['abstract']
    except Exception as e:
        print(f"Error parsing response: {e}")
        return []
