from typing import Dict, Any, List
import json
from pydantic import BaseModel # type: ignore
import os

# 添加项目根目录到sys.path
from pathlib import Path
import sys
project_root = str(Path(__file__).parent.parent.absolute()) # e:\RAG\src\query_transformations\abstractness_analyzer.py\..\..
sys.path.append(project_root)

from model_utils.model_manage import ModelManage

model = ModelManage()

def apply_fuzzy_metadata_filter(collection_name: str, user_filter: Dict[str, Any], metadata_registry_path: str = None) -> Dict[str, Any]:
    """
    将用户输入的 metadata_filter 与本地存储的 metadata 进行模糊匹配，
    返回一个经过调整的 metadata_filter。

    Args:
        user_filter: 用户输入的元数据过滤器，格式为 {key: {operator: [values]}}，类似：{'author': {'$in': ['john', 'jill']}}
        metadata_registry_path: 本地 metadata_registry.json 文件路径

    Returns:
        Dict[str, Any]: 调整后的元数据过滤器
    """
    # 如果没有传入 metadata_registry_path，则使用默认路径
    if metadata_registry_path is None:
        # 获取当前脚本文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建 metadata_registry.json 的绝对路径
        metadata_registry_path = os.path.join(current_dir, "metadata_registry.json")

    # 加载本地 metadata_registry.json
    with open(metadata_registry_path, "r", encoding="utf-8") as f:
        metadata_registry = json.load(f)

    if collection_name not in metadata_registry.keys():
        raise ValueError(f"Collection '{collection_name}' 不存在于 metadata_registry.json 中。")

    metadata_registry = metadata_registry[collection_name]

    # 初始化模糊匹配后的过滤器
    fuzzy_filter = {}

    # 提取有效的 keys 和 values
    valid_keys = list(metadata_registry.keys()) # list
    valid_operators = ["$in", "$eq", "$ne", "$gt", "$gte", "$lt", "$lte"]  # list

    key = list(user_filter.keys())[0]
    operator = list(user_filter[key].keys())[0] # string

    # 调用大模型进行 key 和 operator 的模糊匹配
    matched_key = _fuzzy_match_key(key, valid_keys)
    matched_operator = _fuzzy_match_operator(operator, valid_operators)

    # 获取本地 metadata 中的候选值
    valid_values = metadata_registry[matched_key] # list

    # 提取用户的值
    values = list(user_filter[key].values()) # list

    # 调用大模型进行 operator 和 values 的模糊匹配
    matched_values = _fuzzy_match_values(values, valid_values)

    # 如果有匹配的值，构造新的过滤条件
    if matched_key and matched_operator and matched_values: # 如果匹配到了key、operator和values
        fuzzy_filter[matched_key] = {matched_operator: matched_values}
    else:
        raise ValueError("没有匹配到有效的元数据过滤条件")

    return fuzzy_filter

def _fuzzy_match_key(user_key: str, valid_keys: List[str]) -> str:
    """
    使用大模型的 json mode 对用户输入的 key 进行模糊匹配。

    Args:
        user_key: 用户输入的 key
        valid_keys: 本地 metadata 中的有效 key 列表

    Returns:
        str: 匹配到的 key，如果没有匹配则返回 None
    """
    class FuzzyMatchKey(BaseModel):
        matched_key: str

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "fuzzy_match_key",
            "description": "模糊匹配用户输入的 key 到有效的 metadata key",
            "schema": {
                "type": "object",
                "properties": {
                    "matched_key": {
                        "type": "string",
                        "description": "匹配到的 metadata key",
                        "enum": valid_keys
                    }
                },
                "required": ["matched_key"],
                "additionalProperties": False
            },
            "strict": True
        }
    }

    # 调用大模型进行匹配
    user_prompt = f"请从以下有效的 metadata key 中选择最接近用户输入的一个 key：{user_key}"
    response = model.generate(user_prompt, mode="structured", response_format=response_format) # model.generate 默认的model是支持json mode的
    try:
        temp = FuzzyMatchKey.parse_raw(response).dict()
        return temp["matched_key"]
    except Exception as e:
        print(f"Error parsing response: {e}")
        return None


def _fuzzy_match_operator(user_operator: str, valid_operators: List[str]) -> str:
    """
    使用大模型的 json mode 对用户输入的 operator 进行模糊匹配。

    Args:
        user_operator: 用户输入的 operator
        valid_operators: 预定义的有效操作符列表

    Returns:
        str: 匹配到的 operator，如果没有匹配则返回 None
    """
    class FuzzyMatchOperator(BaseModel):
        matched_operator: str

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "fuzzy_match_operator",
            "description": "模糊匹配用户输入的 operator 到有效的操作符",
            "schema": {
                "type": "object",
                "properties": {
                    "matched_operator": {
                        "type": "string",
                        "description": "匹配到的操作符",
                        "enum": valid_operators
                    }
                },
                "required": ["matched_operator"],
                "additionalProperties": False
            },
            "strict": True
        }
    }

    # 调用大模型进行匹配
    user_prompt = f"请从以下有效的操作符中选择最接近用户输入的一个 operator：{user_operator}"
    response = model.generate(user_prompt, mode="structured", response_format=response_format)
    try:
        temp = FuzzyMatchOperator.parse_raw(response).dict()
        return temp["matched_operator"]
    except Exception as e:
        print(f"Error parsing response: {e}")
        return None


def _fuzzy_match_values(user_values: List[str], valid_values: List[str]) -> List[str]:
    """
    使用大模型的 json mode 对用户输入的值列表进行模糊匹配。

    Args:
        user_values: 用户输入的值列表
        valid_values: 本地 metadata 中的有效值列表

    Returns:
        List[str]: 匹配到的值列表
    """
    class FuzzyMatchValues(BaseModel):
        matched_values: List[str]

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "fuzzy_match_values",
            "description": "模糊匹配用户输入的值到有效的 metadata 值",
            "schema": {
                "type": "object",
                "properties": {
                    "matched_values": {
                        "type": "array",
                        "description": "匹配到的 metadata 值",
                        "items": {
                            "type": "string",
                            "enum": valid_values
                        }
                    }
                },
                "required": ["matched_values"],
                "additionalProperties": False
            },
            "strict": True
        }
    }

    # 调用大模型进行匹配
    user_prompt = f"请从以下有效的 metadata 值中选择最接近用户输入的一个或多个值，这里是用户的输入：{user_values}" 
    response = model.generate(user_prompt, mode="structured", response_format=response_format)
    try:
        temp = FuzzyMatchValues.parse_raw(response).dict()
        return temp["matched_values"]
    except Exception as e:
        print(f"Error parsing response: {e}")
        return []



