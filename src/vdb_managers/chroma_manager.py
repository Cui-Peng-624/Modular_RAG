# 导入标准库
import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv # type: ignore
import chromadb # type: ignore
import chromadb.utils.embedding_functions as embedding_functions # type: ignore
from datetime import datetime
from uuid import uuid4
import numpy as np # type: ignore
<<<<<<< HEAD
from ragatouille import RAGPretrainedModel # type: ignore
=======
>>>>>>> e503d4c6a721d0e7d96523baacc992414b9bc723

# 设置代理
os.environ["http_proxy"] = "127.0.0.1:7897"
os.environ["https_proxy"] = "127.0.0.1:7897"

# 设置client
client = chromadb.PersistentClient(path = "ChromaVDB") # 注意，这里不是就在vdb_managers文件夹下创建永久化chroma数据库，而是这个py文件在哪里调用，就在哪里创建永久化chroma数据库，例如我再tests/chroma_manager_test.ipynb中调用，就在tests文件夹下创建永久化chroma数据库 - 就是在调用这个文件的文件目录下寻找ChromaVDB文件夹，其中就是我们的向量数据库

# 添加项目根目录到sys.path
from pathlib import Path
import sys
project_root = str(Path(__file__).parent.parent.absolute()) # e:\RAG\src\vdb_managers\chroma_manager.py\..\..
sys.path.append(project_root)

# 导入自定义的模块
from document_processors.loader import DocumentLoader
from document_processors.splitter import DocumentSplitter
from sparse_retrievers.bm25_manager import BM25Manager
from model_utils.extract_metadata_sync import extract_metadata_sync
from vdb_managers.fuzzy_metadata_filter import apply_fuzzy_metadata_filter
from indexing.extract_summaries_sync import extract_summaries_sync

class ChromaManager:
    METADATA_REGISTRY_PATH = os.path.join(project_root, 'vdb_managers', 'metadata_registry.json')

    def __init__(self, embedded_model: str = "text-embedding-3-large") -> None:
        # 加载环境变量
        load_dotenv()
        api_key = os.getenv('ZETATECHS_API_KEY')
        base_url = os.getenv('ZETATECHS_API_BASE')

        # 初始化embeddings
        self.openai_embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                api_base=base_url,
                model_name=embedded_model
            )
        
        # 初始化文档切分器
        self.splitter = DocumentSplitter()
        
        # 初始化sparse_retriever
        self.sparse_retriever = BM25Manager()
        
        self._load_metadata_registry() # 每次初始化ChromaManager都会加载现有的全部数据到内存

    def _load_metadata_registry(self):
        """加载或初始化元数据注册表"""
        if os.path.exists(self.METADATA_REGISTRY_PATH):
            with open(self.METADATA_REGISTRY_PATH, 'r', encoding='utf-8') as f:
                self.metadata_registry = json.load(f)
                # 确保所有值是集合类型
                for collection, metadata in self.metadata_registry.items(): # collection就是储存的向量知识库的名称
                    for key, values in metadata.items(): # key是metadata的key，value是一个list
                        self.metadata_registry[collection][key] = set(values)
        else:
            # 初始化为新的结构：每个 collection 有独立的 metadata 键值对
            self.metadata_registry = {}
            self._save_metadata_registry()

    def _save_metadata_registry(self):
        """保存元数据注册表到文件"""
        # 将集合转换为列表以便 JSON 序列化
        serializable_registry = {
            collection: {key: list(values) for key, values in metadata.items()}
            for collection, metadata in self.metadata_registry.items()
        }
        with open(self.METADATA_REGISTRY_PATH, 'w', encoding='utf-8') as f:
            json.dump(serializable_registry, f, ensure_ascii=False, indent=4)

    def _update_metadata_registry(self, collection_name: str, metadatas: List[Dict[str, Any]]): # metadatas类似：[{"category": str, "keywords": str}, {}, ...]
        """
        更新元数据注册表
        Args:
            collection_name: 集合名称
            metadatas: 上传文档的元数据列表
        """
        if collection_name not in self.metadata_registry:
            # 如果 collection 不存在，初始化为一个空字典
            self.metadata_registry[collection_name] = {}

        # 遍历每个 metadata，更新 registry
        for metadata in metadatas:
            for key, value in metadata.items():
                if key not in self.metadata_registry[collection_name]:
                    # 如果 key 不存在，初始化为一个空集合
                    self.metadata_registry[collection_name][key] = set()
                # 将值添加到集合中
                self.metadata_registry[collection_name][key].add(value)

        self._save_metadata_registry()

    def _get_vector_store(self, collection_name: str, discription: str = None, similarity_metric: str = "cosine") -> chromadb.Collection:
        """根据 collection_name 初始化或加载 Chroma 向量存储"""
        metadata = {
            "created_at": str(datetime.now())
        }
        if discription:  # 如果提供了 discription 参数，则添加到 metadata
            metadata["discription"] = discription

        if similarity_metric:
            metadata["hnsw:space"] = similarity_metric

        return client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.openai_embedding_function,
            metadata=metadata
        )
    
<<<<<<< HEAD
    def _upload_documents(self, documents_type: str, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100, auto_extract_metadata: bool = False, metadata: Dict[str, Any] = None, collection_name: str = None, discription: str = None, summarize_chunks: bool = False, similarity_metric: str = 'cosine'):
        """
        上传文件到向量数据库
        Args:
            documents_type: 文档类型，可以是"pdf"或"txt"或"directory"
=======
    def upload_pdf_file(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100, 
                       auto_extract_metadata: bool = False, metadata: Dict[str, Any] = None, 
                       collection_name: str = None, discription: str = None, 
                       summarize_chunks: bool = False, similarity_metric: str = 'cosine') -> None:
        """
        上传PDF文件到向量数据库
        Args:
>>>>>>> e503d4c6a721d0e7d96523baacc992414b9bc723
            file_path: 文件路径
            chunk_size: 每个chunk的最大字符数
            chunk_overlap: 相邻chunk之间的重叠字符数
            auto_extract_metadata：是否使用大模型自动提取每个chunk的特征作为元数据
            metadata: 传入的metadata，如果auto_extract_metadata为False，则使用传入的metadata
            collection_name: 集合名称
            discription: 集合描述
            summarize_chunks: 是否对chunks进行总结
            similarity_metric：l2, ip, consine
        """
        if collection_name is None:
            raise ValueError("你在上传文件的时候必须指定集合名称！")
        
        try:
            documents = DocumentLoader.load_documents(documents_type, file_path)
        except Exception as e:
            raise ValueError(f"加载文档失败: {e}")
        
        splitter = DocumentSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents_chunks = splitter.split_documents(documents)  # 1. list: [Document(metadata={'source': 'files/论文 - GraphRAG.pdf', 'page': 0}, page_content=), Document(), ...] # 2. 通过documents_chunks[i].page_content获取chunk内容

        metadatas = [] # [{"category": str, "keywords": str}, {}, ...]
        
        # 自动提取metadata，只提取category和keywords
        if auto_extract_metadata is True:  # 如果指定要求使用大模型提取元数据，不管有没有传入metadata，我们都不管
            extracted_metadata = extract_metadata_sync(file_path, documents_chunks)  # list: [{'category': '***', 'keywords': ['', '', '', '', '']}, {}, ...]
            for item in extracted_metadata: 
                category, keywords = item["category"], item["keywords"] # str, list[str]
                metadatas.append({"category": category, "keywords": keywords[0]}) # 只添加第一个关键词，因为chroma的metadata要求的value不能是list
        elif auto_extract_metadata is False and metadata is not None:  # 传入的metadata的格式也应该是dict
            metadatas = [metadata.copy() for _ in range(len(documents_chunks))]
        else: # auto_extract_metadata is False and metadata is None
            metadatas = [{"category": "default", "keywords": "default"} for _ in range(len(documents_chunks))]

        documents = [chunk.page_content for chunk in documents_chunks]

        # 如果需要对chunk进行总结，则documents=summaries，原始文档信息存到metadata中
        # 如果不需要总结，metadata中不存在original_text这个key值
        if summarize_chunks:
            # 使用LLM对每个chunk进行总结
            summaries = extract_summaries_sync(documents)  # 异步获取总结
            # 将原始文档存入metadata，总结作为document
            for i, metadata in enumerate(metadatas):
                metadata["original_text"] = documents[i]  # 原始文档存入metadata
            # 使用总结作为document
            documents = summaries

        #####################################################################################
        if summarize_chunks:
            # 使用LLM对每个chunk进行总结
            summaries = extract_summaries_sync(documents)  # 异步获取总结
            
            # 将原始文档存入metadata，总结作为document
            for i, metadata in enumerate(metadatas):
                metadata["original_text"] = documents[i]  # 原始文档存入metadata
            
            # 使用总结作为document
            documents = summaries
        #####################################################################################

        # 初始化对应的 vector_store
        vector_store = self._get_vector_store(collection_name=collection_name, 
<<<<<<< HEAD
                                              discription=discription, 
                                              similarity_metric=similarity_metric)
=======
                                            discription=discription, 
                                            similarity_metric=similarity_metric)

        # 更新元数据注册表 - 我们需要记录的是什么？是每个collection下面每个chunk的关键字，方便后续当我们需要使用关键字查询的时候进行匹配
        self._update_metadata_registry(collection_name, metadatas) # (str, list(dict))
>>>>>>> e503d4c6a721d0e7d96523baacc992414b9bc723

        # 将原始文档信息存入metadata
        for metadata, documents_chunk in zip(metadatas, documents_chunks):
            metadata["source"] = documents_chunk.metadata.get("source", None)
            metadata["page"] = documents_chunk.metadata.get("page", None)
        
        # 更新元数据注册表 - 我们需要记录的是什么？是每个collection下面每个chunk的关键字，方便后续当我们需要使用关键字查询的时候进行匹配
        self._update_metadata_registry(collection_name, metadatas) # (str, list(dict))

        # 最后再更新一下uuid
        uuids = [str(uuid4()) for _ in range(len(documents))]

        # print(len(documents), len(metadatas), len(uuids))
        # 将数据存入向量数据库
        vector_store.add(documents=documents, metadatas=metadatas, ids=uuids)

        # 更新 BM25 索引和文档内容
        self.sparse_retriever.add_documents(collection_name, documents)



    def upload_pdf_file(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100, auto_extract_metadata: bool = False, metadata: Dict[str, Any] = None, collection_name: str = None, discription: str = None, summarize_chunks: bool = False, similarity_metric: str = 'cosine') -> None:
        self._upload_documents("pdf", file_path, chunk_size, chunk_overlap, auto_extract_metadata, metadata, collection_name, discription, summarize_chunks, similarity_metric)
        # """
        # 上传PDF文件到向量数据库
        # Args:
        #     file_path: 文件路径
        #     chunk_size: 每个chunk的最大字符数
        #     chunk_overlap: 相邻chunk之间的重叠字符数
        #     auto_extract_metadata：是否使用大模型自动提取每个chunk的特征作为元数据
        #     metadata: 传入的metadata，如果auto_extract_metadata为False，则使用传入的metadata
        #     collection_name: 集合名称
        #     discription: 集合描述
        #     summarize_chunks: 是否对chunks进行总结
        #     similarity_metric：l2, ip, consine
        # """
        # if collection_name is None:
        #     raise ValueError("你在上传文件的时候必须指定集合名称！")

        # # 加载和切分文档
        # documents = DocumentLoader.load_pdf(file_path)
        # splitter = DocumentSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # documents_chunks = splitter.split_documents(documents)  # 1. list: [Document(metadata={'source': 'files/论文 - GraphRAG.pdf', 'page': 0}, page_content=), Document(), ...] # 2. 通过documents_chunks[i].page_content获取chunk内容
        
        # metadatas = [] # [{"category": str, "keywords": str}, {}, ...]
        
        # # 自动提取metadata，只提取category和keywords
        # if auto_extract_metadata is True:  # 如果指定要求使用大模型提取元数据，不管有没有传入metadata，我们都不管
        #     extracted_metadata = extract_metadata_sync(file_path, documents_chunks)  # list: [{'category': '***', 'keywords': ['', '', '', '', '']}, {}, ...]
        #     for item in extracted_metadata: 
        #         category, keywords = item["category"], item["keywords"] # str, list[str]
        #         metadatas.append({"category": category, "keywords": keywords[0]}) # 只添加第一个关键词，因为chroma的metadata要求的value不能是list
        # elif auto_extract_metadata is False and metadata is not None:  # 传入的metadata的格式也应该是dict
        #     metadatas = [metadata.copy() for _ in range(len(documents_chunks))]
        # else: # auto_extract_metadata is False and metadata is None
        #     metadatas = [{"category": "default", "keywords": "default"} for _ in range(len(documents_chunks))]

        # documents = [chunk.page_content for chunk in documents_chunks]

        # # 如果需要对chunk进行总结，则documents=summaries，原始文档信息存到metadata中
        # # 如果不需要总结，metadata中不存在original_text这个key值
        # if summarize_chunks:
        #     # 使用LLM对每个chunk进行总结
        #     summaries = extract_summaries_sync(documents)  # 异步获取总结
        #     # 将原始文档存入metadata，总结作为document
        #     for i, metadata in enumerate(metadatas):
        #         metadata["original_text"] = documents[i]  # 原始文档存入metadata
        #     # 使用总结作为document
        #     documents = summaries

        # # 初始化对应的 vector_store
        # vector_store = self._get_vector_store(collection_name=collection_name, 
        #                                       discription=discription, 
        #                                       similarity_metric=similarity_metric)

        # # 将原始文档信息存入metadata
        # for metadata, documents_chunk in zip(metadatas, documents_chunks):
        #     metadata["source"] = documents_chunk.metadata.get("source", None)
        #     metadata["page"] = documents_chunk.metadata.get("page", None)
        
        # # 更新元数据注册表 - 我们需要记录的是什么？是每个collection下面每个chunk的关键字，方便后续当我们需要使用关键字查询的时候进行匹配
        # self._update_metadata_registry(collection_name, metadatas) # (str, list(dict))

        # # 最后再更新一下uuid
        # uuids = [str(uuid4()) for _ in range(len(documents))]

        # # print(len(documents), len(metadatas), len(uuids))
        # # 将数据存入向量数据库
        # vector_store.add(documents=documents, metadatas=metadatas, ids=uuids)

        # # 更新 BM25 索引和文档内容
        # self.sparse_retriever.add_documents(collection_name, documents)

    def upload_txt_file(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100, auto_extract_metadata: bool = False, metadata: Dict[str, Any] = None, collection_name: str = None, discription: str = None, summarize_chunks: bool = False, similarity_metric: str = 'cosine') -> None:
        """上传TXT文件到向量数据库并提取元数据"""
        self._upload_documents("txt", file_path, chunk_size, chunk_overlap, auto_extract_metadata, metadata, collection_name, discription, summarize_chunks, similarity_metric)

    def upload_directory(self, directory_path: str, chunk_size: int = 1000, chunk_overlap: int = 100, auto_extract_metadata: bool = False, metadata: Dict[str, Any] = None, collection_name: str = None, discription: str = None, summarize_chunks: bool = False, similarity_metric: str = 'cosine') -> None:
        """上传目录中的所有PDF和TXT文件到向量数据库并提取元数据"""
        self._upload_documents("directory", directory_path, chunk_size, chunk_overlap, auto_extract_metadata, metadata, collection_name, discription, summarize_chunks, similarity_metric)

    def dense_search(self, collection_name: str, query: str, k: int = 3, 
                    metadata_filter: Dict[str, Any] = None, fuzzy_filter: bool = False,
<<<<<<< HEAD
                    use_summary: bool = False, 
                    colbert_rerank: bool = False, colbert_index_name: str = None, colbert_max_document_length: int = None, colbert_split_documents: bool = True, colbert_k: int = 1) -> List[Dict[str, Any]]:
=======
                    use_summary: bool = False) -> List[Dict[str, Any]]:
>>>>>>> e503d4c6a721d0e7d96523baacc992414b9bc723
        """
        密集向量搜索
        Args:
            collection_name: 集合名称
            query: 查询文本
            k: 返回结果数量
<<<<<<< HEAD
            metadata_filter: 元数据过滤条件。metadata_filter example: metadata_filter = {'author': {'$in': ['john', 'jill']}}
            fuzzy_filter: 是否启用模糊元数据过滤
            use_summary: 是否返回总结内容。True返回总结，False返回原文。一般只有在upload的时候指定了需要使用summary，所以dense_search中我们认为用户仅会在upload时使用了summary这里才会考虑返回的是summary还是original chunks。
            # search_direction: 检索方向，可选值为"t2d"（自定向下），"d2t"（自下向上）。
            # max_level: 在t2d模式下，表示从level等于多少开始往下搜索。在d2t模式下，表示从level等于多少开始往上搜索。
            # include_children: 是否包含子文档，仅在search_direction为"d2t"时有效
            # include_parents: 是否包含父文档，仅在search_direction为"d2t"时有效
            colbert_rerank: 是否使用ColBERT进行重排序
            colbert_index_name: ColBERT的索引名称
            colbert_max_document_length: ColBERT的每个chunk的最大字符数
            colbert_split_documents: 是否对文档进行切分
            colbert_k: ColBERT的返回结果数量
=======
            metadata_filter: 元数据过滤条件
                metadata_filter example: metadata_filter = {'author': {'$in': ['john', 'jill']}}
            fuzzy_filter: 是否启用模糊元数据过滤
            use_summary: 是否返回总结内容。True返回总结，False返回原文
>>>>>>> e503d4c6a721d0e7d96523baacc992414b9bc723
        """
        # 加载向量数据库
        vector_store = self._get_vector_store(collection_name=collection_name)

        # 如果启用了模糊元数据过滤
        if fuzzy_filter and metadata_filter: # 如果启用了模糊元数据过滤，并且传入了metadata_filter
            metadata_filter = apply_fuzzy_metadata_filter(collection_name, metadata_filter)
            # print("模糊元数据过滤后的过滤器：", metadata_filter, "\n")

        # 获取检索结果
        results = vector_store.query(
            query_texts=[query],
            where=metadata_filter,
            n_results=k,
            include=["metadatas", "documents", "distances"]
        )

        # 如果不使用summary（即需要原始文档），则从metadata中获取原始文档
        if not use_summary and results["metadatas"]: # 如果use_summary为False，并且检索结果中存在metadata
            for i, metadata in enumerate(results["metadatas"][0]):
<<<<<<< HEAD
                if "original_text" in metadata.keys(): # 交换一下
                    temp = metadata["original_text"]
                    metadata["summarized_text"] = results["documents"][0][i]
                    results["documents"][0][i] = temp
                    del metadata["original_text"]
                else: # 存在一类情况，upload的时候有时候不需要summary，但检索的时候一起检索，所以可能会有的chunk的metadata有original_text，有的没有，所以需要边缘处理 - 这里仅仅是为了不报错
                    continue

        # 如果需要使用ColBERT进行重排序 
        if colbert_rerank:
            if k > colbert_k: 
                RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
                RAG.index(
                    collection=results["documents"][0], # list[str]
                    document_metadatas=results["metadatas"][0], # list[dict]
                    index_name=colbert_index_name,
                    max_document_length=colbert_max_document_length,
                    split_documents=colbert_split_documents,
                )
            else:
                raise ValueError("colbert_k的值必须小于k的值！")
            
            # 使用ColBERT进行重排序
            results = RAG.search(query=query, k=colbert_k)

            # 转换格式
            formatted_results = {
                'ids': [[result['document_id'] for result in results]],
                'documents': [[result['content'] for result in results]],
                'metadatas': [[result["document_metadata"] for result in results]], # 不用担心document_metadata为空，因为chroma在上传的时候必定会有metadata
                'distances': [[1 - result['score']/100 for result in results]]  # 将score转换为distance
            }
    
            results = formatted_results
=======
                if "original_text" in metadata:
                    results["documents"][0][i] = metadata["original_text"]
                else: # 存在一类情况，upload的时候有时候不需要summary，但检索的时候一起检索，所以可能会有的chunk的metadata有original_text，有的没有，所以需要边缘处理
                    results["documents"][0][i] = results["documents"][0][i]
>>>>>>> e503d4c6a721d0e7d96523baacc992414b9bc723

        return results
    
    def _get_metadata_from_vector_store(self, collection_name: str, document: str) -> Dict[str, Any]:
        """
        从向量数据库中获取指定文档的 metadata
        """
        vector_store = self._get_vector_store(collection_name=collection_name)
        results = vector_store.query(
            query_texts=[document],  # 使用文档内容作为查询
            n_results=1,  # 只需要返回一个匹配结果
            include=["metadatas"]  # 只需要 metadata # include参数，返回的是啥，这里只要求返回metadatas
        )
        if results and results.get("metadatas", [])[0]:
            return results["metadatas"][0][0]  # 返回第一个匹配的 metadata
        return {}

    def hybrid_search(self, collection_name: str, query: str, k: int = 3, dense_weight: float = 0.5, metadata_filter: Dict[str, Any] = None, fuzzy_filter: bool = False, use_summary: bool = False, colbert_rerank: bool = False, colbert_index_name: str = None, colbert_max_document_length: int = None, colbert_split_documents: bool = True, colbert_k: int = 1) -> List[Dict[str, Any]]:
        """混合搜索方法，支持元数据过滤，支持模糊元数据过滤"""
        # 加载稀疏索引
        self.sparse_retriever.load_collection(collection_name)

        # 如果启用了模糊元数据过滤
        if fuzzy_filter and metadata_filter:
            metadata_filter = apply_fuzzy_metadata_filter(collection_name, metadata_filter)
            # print("模糊元数据过滤后的过滤器：", metadata_filter, "\n")

        # 1. 获取密集向量搜索结果
        dense_results = self.dense_search(collection_name = collection_name, query = query, k=k, metadata_filter=metadata_filter, fuzzy_filter=fuzzy_filter, use_summary=use_summary, colbert_rerank=colbert_rerank, colbert_index_name=colbert_index_name, colbert_max_document_length=colbert_max_document_length, colbert_split_documents=colbert_split_documents, colbert_k=colbert_k)
        dense_documents = dense_results.get('documents', [])[0]  # [[]]，[0]代表返回[]
        dense_metadatas = dense_results.get('metadatas', [])[0]
        dense_distances = dense_results.get('distances', [])[0]

        # 将密集搜索结果转换为字典
        dense_dict = {
            doc: {
                'score': 1 - distance,  # chroma的cosine计算是"1-余弦相似度"
                'metadata': metadata,
                'content': doc
            }
            for doc, metadata, distance in zip(dense_documents, dense_metadatas, dense_distances)
        }

        # 2. 获取稀疏搜索结果
        sparse_results = self.sparse_retriever.search(query, k=k)
        sparse_dict = {
            result['content']: {
                'score': result['score'],
                'content': result['content']
            }
            for result in sparse_results
        }

        # 3. 为稀疏搜索结果补充 metadata
        for doc in sparse_dict.keys():
            if doc not in dense_dict:  # 如果稀疏搜索的文档不在密集搜索结果中
                # 从向量数据库中查询 metadata
                metadata = self._get_metadata_from_vector_store(collection_name, doc)
                sparse_dict[doc]['metadata'] = metadata

        # 4. 融合结果
        all_docs = set(dense_dict.keys()) | set(sparse_dict.keys())
        hybrid_results = []

        for doc in all_docs:
            hybrid_score = 0.0
            metadata = {}

            if doc in dense_dict:
                hybrid_score += dense_weight * dense_dict[doc]['score']
                metadata = dense_dict[doc].get('metadata', {})

            if doc in sparse_dict:
                hybrid_score += (1 - dense_weight) * sparse_dict[doc]['score']
                # 如果 metadata 为空，尝试从稀疏搜索结果中获取
                if not metadata:
                    metadata = sparse_dict[doc].get('metadata', {})

            # 如果有元数据过滤条件，检查是否符合
            if metadata_filter:
                if not all(metadata.get(key) == value for key, value in metadata_filter.items()):
                    continue

            hybrid_results.append({
                'content': doc,
                'metadata': metadata,
                'score': hybrid_score,
            })

        # 按分数降序排序并返回前 k 个结果
        return sorted(hybrid_results, key=lambda x: x['score'], reverse=True)[:k]

    # 综合了上述的两种搜索模式，根据dense_weight的大小选择不同的搜索方式，返回list[str]
    # 虽然貌似跟hybrid_search很像，。。。
    """
    self, collection_name: str, query: str, k: int = 3, dense_weight: float = 0.5, metadata_filter: Dict[str, Any] = None, fuzzy_filter: bool = False, use_summary: bool = False, colbert_rerank: bool = False, colbert_index_name: str = None, colbert_max_document_length: int = None, colbert_split_documents: bool = True, colbert_k: int = 1
    
    """
    def search(self, collection_name: str = None, query: str = None, k: int = 3, dense_weight: float = 0.5, metadata_filter: Dict[str, Any] = None, fuzzy_filter: bool = False, use_summary: bool = False, colbert_rerank: bool = False, colbert_index_name: str = None, colbert_max_document_length: int = None, colbert_split_documents: bool = True, colbert_k: int = 1, **kwargs) -> list[dict]:
        """
        根据指定的搜索模式和元数据过滤获取检索结果并返回内容列表

        Args:
            query: 查询文本
            k: 返回的结果数量
            metadata_filter: 元数据过滤条件，格式参考 Chroma 的 metadata filtering 文档
            dense_weight: 混合搜索模式中的密集搜索权重，默认值为 0.5，注意，此参数还可以用于表示单独的系数搜索和混合搜索
            collection_name: 集合名称
            fuzzy_filter: 是否启用模糊元数据过滤
            **kwargs:

        Returns:
            list[dict]: 检索到的文档内容列表

        Raises:
            ValueError: 当指定的 mode 不支持时抛出异常
        """
        if collection_name:
            vector_store = self._get_vector_store(collection_name=collection_name)
        else:
            raise ValueError("您输入的集合名称错误！")
        
        # 检查dense_weight是否在0-1之间
        if not 0 <= dense_weight <= 1:
            raise ValueError("dense_weight的范围是0-1。0代表仅使用稀疏搜索，1代表仅使用密集搜索。")

        # 根据dense_weight选择不同的搜索模式
        results = self.hybrid_search(
            collection_name=collection_name,
            query=query,
            k=k,
            dense_weight=dense_weight,
            metadata_filter=metadata_filter,
            fuzzy_filter=fuzzy_filter,
            use_summary=use_summary,
            colbert_rerank=colbert_rerank,
            colbert_index_name=colbert_index_name,
            colbert_max_document_length=colbert_max_document_length,
            colbert_split_documents=colbert_split_documents,
            colbert_k=colbert_k
        )

        return results

    # 获取格式化的上下文字符串。此函数会先调用search获取内容列表，
    # 然后将其转换为格式化的字符串（就是拼接在一起）。
    def get_formatted_context(self, query: str, k: int = 3, metadata_filter: Dict[str, Any] = None, dense_weight: float = 0.5, collection_name: str = None, fuzzy_filter: bool = False, use_summary: bool = False, **kwargs) -> str:
        """
        获取格式化的上下文字符串。此函数会先调用search获取内容列表，
        然后将其转换为格式化的字符串。
        
        Args:
            query: 查询文本
            mode: 搜索模式，可选值为 "hybrid", "dense"
            k: 返回的结果数量
            metadata_filter: 元数据过滤条件
            dense_weight: 混合搜索模式中的密集搜索权重，默认值为 0.5
            collection_name: 集合名称
            fuzzy_filter: 是否启用模糊元数据过滤
            **kwargs: 其他参数，比如 hybrid_search 的额外参数
            
        Returns:
            str: 格式化后的上下文字符串
        """
        # 获取内容列表
        results = self.search(
            query=query,
            k=k,
            metadata_filter=metadata_filter,
            dense_weight=dense_weight,
            collection_name=collection_name,
            fuzzy_filter=fuzzy_filter,
            use_summary=use_summary,
            **kwargs
        )
        
        # 如果没有检索到内容，返回提示信息
        if not results:
            return "未检索到相关内容。"

        # 格式化每个文档片段
        formatted_chunks = []
        for i, result in enumerate(results, 1):
            # 清理文本（移除多余的空白字符）
            cleaned_content = " ".join(result["content"].split())
            # 添加编号和格式化
            formatted_chunk = f"[文档片段 {i}]\n{cleaned_content}"
            formatted_chunks.append(formatted_chunk)
        
        # 使用分隔线连接所有文档片段
        formatted_context = "\n\n---\n\n".join(formatted_chunks)
        
        return formatted_context

############################################################################################
############################################################################################
############################################################################################

"""
ChromaManager类总结：
包含：
    _load_metadata_registry
    _save_metadata_registry
    _update_metadata_registry
    _get_vector_store
    upload_pdf_file
    dense_search
    _get_metadata_from_vector_store
    hybrid_search
    search
    get_formatted_context
"""

from indexing.raptor_config import RaptorConfig
from indexing.raptor_manager import RaptorManager

# RAPTOR功能扩展
# RAPTOR是一个较为复杂的层次化检索系统，为避免代码冗余，这里将RAPTOR的实现与ChromaManager分开
# ChromaManager_RAPTOR继承自ChromaManager，并实现了RAPTOR的的上传和检索这两大功能
class ChromaManager_RAPTOR(ChromaManager):
    """集成RAPTOR功能的管理器"""
    def __init__(self, n_levels: int = 2, min_cluster_size: int = 3):
        """
        n_levels: 聚类几次
        min_cluster_size: 最小聚类大小：如果一个聚类中的文档数量小于这个值，则该聚类将被合并到其他聚类中。
        """
        super().__init__()
        self.n_levels = n_levels
        self.min_cluster_size = min_cluster_size
        self.raptor_config = RaptorConfig(enabled=True, n_levels=self.n_levels, min_cluster_size=self.min_cluster_size) # enabled必须为True，否则RAPTOR功能不会生效
        self.raptor_manager = RaptorManager(self.raptor_config)

        # 初始化sparse_retriever
        self.sparse_retriever = BM25Manager()

    def upload_documents(self, documents_type: str, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100, metadata: Dict[str, Any] = None, collection_name: str = None, discription: str = None, similarity_metric: str = "cosine") -> None:
        if collection_name is None:
            raise ValueError("你在上传文件的时候必须指定集合名称！")

        try:
            documents = DocumentLoader.load_documents(documents_type, file_path)
        except Exception as e:
            raise ValueError(f"加载文档失败: {e}")

        splitter = DocumentSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents_chunks = splitter.split_documents(documents)  # 1. list: [Document(metadata={'source': 'files/论文 - GraphRAG.pdf', 'page': 0}, page_content='', Document(), ...] # 2. 通过documents_chunks[i].page_content获取chunk内容
        documents = [chunk.page_content for chunk in documents_chunks]
        
        metadatas = [] # [{"category": str, "keywords": str}, {}, ...]

        try:
            # 使用RAPTOR构建层次结构 - hierarchy的具体形式请参考：test_raptor_integration.ipynb
            hierarchy = self.raptor_manager.build_hierarchy(documents)
            
            if hierarchy: # 如果hierarchy不为空
                all_docs = []
                all_metadatas = []
                processed_docs = set()
                
                # 使用doc_id_map确保文档ID的一致性
                doc_id_map = hierarchy[0]['doc_id_map'] # doc_id_map是一个全局的信息，包含在RAPTOR过程中所有id和chunk的对应。每一层的doc_id_map都是一样的。
                
                # 构建parent_docs映射 
                parent_docs_map = {} # 找到每个子文档对应的父文档
                for level_data in hierarchy.values(): # n_levels个list
                    for summary_id, child_ids in level_data['parent_child_map'].items(): # summary_id：str，child_ids：list[str]
                        for child_id in child_ids:
                            if child_id not in parent_docs_map: # 如果子文档还没有父文档
                                parent_docs_map[child_id] = []
                            parent_docs_map[child_id].append(summary_id) # 将子文档的id和父文档的id对应起来
                
                # 处理每个层级的文档
                for level, level_data in hierarchy.items():
                    # 处理原始文档
                    for doc_id in level_data['doc_ids']: # doc_ids里面只是这一level的文档id，不包含父文档或子文档的id
                        if doc_id not in processed_docs:
                            doc = doc_id_map[doc_id]
                            doc_metadata = metadata.copy() if metadata else {}
                            doc_metadata.update({
                                "doc_id": doc_id,
                                "level": level,
                                # "is_summary": False,
                                "file_path": file_path,
                                "parent_docs": json.dumps(parent_docs_map.get(doc_id, []))
                            })
                            
                            all_docs.append(doc)
                            all_metadatas.append(doc_metadata)
                            processed_docs.add(doc_id)
                    
                    # 处理摘要文档
                    for summary_id in level_data['summary_ids']: # summary_ids里只包含这一level的父文档的id
                        if summary_id not in processed_docs:
                            summary = doc_id_map[summary_id]
                            summary_metadata = metadata.copy() if metadata else {}
                            summary_metadata.update({
                                "doc_id": summary_id,
                                "level": level + 1,
                                # "is_summary": True,
                                "file_path": file_path,
                                "child_docs": json.dumps(level_data['parent_child_map'][summary_id]),
                                "parent_docs": json.dumps(parent_docs_map.get(summary_id, []))
                            })
                            
                            all_docs.append(summary)
                            all_metadatas.append(summary_metadata)
                            processed_docs.add(summary_id)
                
                documents = all_docs
                metadatas = all_metadatas

        except Exception as e:
            print(f"RAPTOR处理错误: {str(e)}")
            return None

        # 初始化对应的 vector_store
        vector_store = self._get_vector_store(collection_name=collection_name, 
                                              discription=discription, 
                                              similarity_metric=similarity_metric)

        # 生成UUID
        uuids = [str(uuid4()) for _ in range(len(documents))]

        # 将原始文档信息存入metadata
        for metadata, documents_chunk in zip(metadatas, documents_chunks):
            metadata["source"] = documents_chunk.metadata.get("source", None)
            metadata["page"] = documents_chunk.metadata.get("page", None)
            
        # 更新元数据注册表
        self._update_metadata_registry(collection_name, metadatas)

        # 将数据存入向量数据库
        print(type(documents), len(documents), type(metadatas), len(metadatas), type(uuids), len(uuids))
        vector_store.add(documents=documents, metadatas=metadatas, ids=uuids)

        # 更新 BM25 索引和文档内容
        self.sparse_retriever.add_documents(collection_name, documents)


    def upload_pdf_file(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100, metadata: Dict[str, Any] = None, collection_name: str = None, discription: str = None, similarity_metric: str = "cosine") -> None:
        """上传PDF文件到向量数据库并提取元数据"""
        self._upload_documents("pdf", file_path, chunk_size, chunk_overlap, metadata, collection_name, discription, similarity_metric)
        # """
        # 重写上传方法，集成RAPTOR的层次化处理
        # """
        # if collection_name is None:
        #     raise ValueError("你在上传文件的时候必须指定集合名称！")

        # # 加载和切分文档
        # documents = DocumentLoader.load_pdf(file_path)
        # splitter = DocumentSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # documents_chunks = splitter.split_documents(documents)  # 1. list: [Document(metadata={'source': 'files/论文 - GraphRAG.pdf', 'page': 0}, page_content='', Document(), ...] # 2. 通过documents_chunks[i].page_content获取chunk内容
        # documents = [chunk.page_content for chunk in documents_chunks]
        
        # metadatas = [] # [{"category": str, "keywords": str}, {}, ...]

        # try:
        #     # 使用RAPTOR构建层次结构 - hierarchy的具体形式请参考：test_raptor_integration.ipynb
        #     hierarchy = self.raptor_manager.build_hierarchy(documents)
            
        #     if hierarchy: # 如果hierarchy不为空
        #         all_docs = []
        #         all_metadatas = []
        #         processed_docs = set()
                
        #         # 使用doc_id_map确保文档ID的一致性
        #         doc_id_map = hierarchy[0]['doc_id_map'] # doc_id_map是一个全局的信息，包含在RAPTOR过程中所有id和chunk的对应。每一层的doc_id_map都是一样的。
                
        #         # 构建parent_docs映射 
        #         parent_docs_map = {} # 找到每个子文档对应的父文档
        #         for level_data in hierarchy.values(): # n_levels个list
        #             for summary_id, child_ids in level_data['parent_child_map'].items(): # summary_id：str，child_ids：list[str]
        #                 for child_id in child_ids:
        #                     if child_id not in parent_docs_map: # 如果子文档还没有父文档
        #                         parent_docs_map[child_id] = []
        #                     parent_docs_map[child_id].append(summary_id) # 将子文档的id和父文档的id对应起来
                
        #         # 处理每个层级的文档
        #         for level, level_data in hierarchy.items():
        #             # 处理原始文档
        #             for doc_id in level_data['doc_ids']: # doc_ids里面只是这一level的文档id，不包含父文档或子文档的id
        #                 if doc_id not in processed_docs:
        #                     doc = doc_id_map[doc_id]
        #                     doc_metadata = metadata.copy() if metadata else {}
        #                     doc_metadata.update({
        #                         "doc_id": doc_id,
        #                         "level": level,
        #                         # "is_summary": False,
        #                         "file_path": file_path,
        #                         "parent_docs": json.dumps(parent_docs_map.get(doc_id, []))
        #                     })
                            
        #                     all_docs.append(doc)
        #                     all_metadatas.append(doc_metadata)
        #                     processed_docs.add(doc_id)
                    
        #             # 处理摘要文档
        #             for summary_id in level_data['summary_ids']: # summary_ids里只包含这一level的父文档的id
        #                 if summary_id not in processed_docs:
        #                     summary = doc_id_map[summary_id]
        #                     summary_metadata = metadata.copy() if metadata else {}
        #                     summary_metadata.update({
        #                         "doc_id": summary_id,
        #                         "level": level + 1,
        #                         # "is_summary": True,
        #                         "file_path": file_path,
        #                         "child_docs": json.dumps(level_data['parent_child_map'][summary_id]),
        #                         "parent_docs": json.dumps(parent_docs_map.get(summary_id, []))
        #                     })
                            
        #                     all_docs.append(summary)
        #                     all_metadatas.append(summary_metadata)
        #                     processed_docs.add(summary_id)
                
        #         documents = all_docs
        #         metadatas = all_metadatas

        # except Exception as e:
        #     print(f"RAPTOR处理错误: {str(e)}")
        #     return None

        # # 初始化对应的 vector_store
        # vector_store = self._get_vector_store(collection_name=collection_name, 
        #                                       discription=discription, 
        #                                       similarity_metric=similarity_metric)

        # # 生成UUID
        # uuids = [str(uuid4()) for _ in range(len(documents))]

        # # 将原始文档信息存入metadata
        # for metadata, documents_chunk in zip(metadatas, documents_chunks):
        #     metadata["source"] = documents_chunk.metadata.get("source", None)
        #     metadata["page"] = documents_chunk.metadata.get("page", None)
            
        # # 更新元数据注册表
        # self._update_metadata_registry(collection_name, metadatas)

        # # 将数据存入向量数据库
        # print(type(documents), len(documents), type(metadatas), len(metadatas), type(uuids), len(uuids))
        # vector_store.add(documents=documents, metadatas=metadatas, ids=uuids)

        # # 更新 BM25 索引和文档内容
        # self.sparse_retriever.add_documents(collection_name, documents)

    def upload_txt_file(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100, metadata: Dict[str, Any] = None, collection_name: str = None, discription: str = None, similarity_metric: str = "cosine") -> None:
        """上传TXT文件到向量数据库并提取元数据"""
        self._upload_documents("txt", file_path, chunk_size, chunk_overlap, metadata, collection_name, discription, similarity_metric)

    def upload_directory(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100, metadata: Dict[str, Any] = None, collection_name: str = None, discription: str = None, similarity_metric: str = "cosine") -> None:
        """上传目录中的所有PDF和TXT文件到向量数据库并提取元数据"""
        self._upload_documents("directory", file_path, chunk_size, chunk_overlap, metadata, collection_name, discription, similarity_metric)

    def _get_vector_store(self, collection_name: str, discription: str = None, similarity_metric: str = "cosine") -> chromadb.Collection:
        """根据 collection_name 初始化或加载 Chroma 向量存储"""
        metadata = {
            "created_at": str(datetime.now())
        }
        if discription:  # 如果提供了 discription 参数，则添加到 metadata
            metadata["discription"] = discription

        if similarity_metric:
            metadata["hnsw:space"] = similarity_metric

        return client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.openai_embedding_function,
            metadata=metadata
        )
    
    def format_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将Chroma的搜索结果转换为标准格式
        
        Args:
            results: Chroma原始搜索结果，格式为：
            [{
                'ids': [[id1, id2, ...]],
                'documents': [[doc1, doc2, ...]],
                'metadatas': [[metadata1, metadata2, ...]],
                'distances': [[distance1, distance2, ...]]
            }, ...]
            
        Returns:
            格式化后的结果列表：
            [{
                'content': str,
                'metadata': Dict[str, Any],
                'score': float
            }, ...]
        """
        formatted_results = []
        
        # 遍历每个结果组
        for result in results:
            # 确保结果包含必要的字段
            if not all(key in result for key in ['documents', 'metadatas', 'distances']):
                continue
                
            # 遍历每组中的单个结果
            for doc, metadata, distance in zip(
                result['documents'][0],  # 第一个列表包含文档内容
                result['metadatas'][0],  # 第一个列表包含元数据
                result['distances'][0]    # 第一个列表包含距离值
            ):
                formatted_results.append({
                    'content': doc,
                    'metadata': metadata,
                    'score': 1 - distance  # 将距离转换为相似度分数
                })
        
        return formatted_results

    def base_search(self, collection_name: str, query: str, k: int = 3, metadata_filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        vector_store = self._get_vector_store(collection_name=collection_name)

        results = vector_store.query(
            query_texts=[query],
            where=metadata_filter,
            n_results=k,
            include=["metadatas", "documents", "distances"]
        )

        return results
        

    def search(self, collection_name: str, query: str, k: int = 3, metadata_filter: Dict[str, Any] = None, search_type: str = None, d: int = 3, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        collection_name: 集合名称
        query: 查询文本
        k: 返回的结果数量
        metadata_filter: 元数据过滤条件
        search_type: traversal 或 collapsed
        d: traversal 检索模式下的检索深度。n_levels代表聚类几次，所以总共是n_levels+1个层次（从1开始）。
        top_k: traversal 检索模式下每个层次检索的文档数量
        """

        # 加载向量数据库
        vector_store = self._get_vector_store(collection_name=collection_name)

        if search_type == "traversal":
            final_results = []
            current_child_ids = set()  # 当前这一级别的节点的子节点，这里仅仅是初始化，防止后面报错使用的
            for level in reversed(range(max(0, self.n_levels-d), self.n_levels+1)): # 从n_levels开始
                # print(level, current_child_ids)
                level_filter = {'level': {'$eq': level}}
                if current_child_ids:
                    parent_filter = {"doc_id": {'$in': list(current_child_ids)}}
                    level_filter = {'$and': [level_filter, parent_filter]}
                # print(level_filter)
                results = self.base_search(collection_name, query, top_k, level_filter)
                # print("here")
                if results['documents']:
                    # 更新current_child_ids为当前层级找到的文档id
                    current_child_ids = set()
                    for metadata in results['metadatas'][0]:  # results['metadatas']是二维列表
                        if 'child_docs' in metadata.keys():
                            child_docs = json.loads(metadata.get('child_docs'))
                            current_child_ids.update(child_docs)
                    final_results.append(results)
            return self.format_results(final_results)
        elif search_type == "collapsed":
            results = self.base_search(collection_name, query, k, metadata_filter)
            return self.format_results([results])
        else:
            raise ValueError("search_type必须为traversal或collapsed")
        
    # 获取格式化的上下文字符串。此函数会先调用search获取内容列表，
    # 然后将其转换为格式化的字符串（就是拼接在一起）。
    def get_formatted_context(self, collection_name: str, query: str, k: int = 3, metadata_filter: Dict[str, Any] = None, search_type: str = None, d: int = 3, top_k: int = 1) -> str:
        """
        获取格式化的上下文字符串。此函数会先调用search获取内容列表，
        然后将其转换为格式化的字符串。
        
        Args:
            collection_name: 集合名称
            query: 查询文本
            k: 返回的结果数量
            metadata_filter: 元数据过滤条件
            search_type: traversal 或 collapsed
            d: traversal 检索模式下的检索深度。n_levels代表聚类几次，所以总共是n_levels+1个层次（从1开始）。
            top_k: traversal 检索模式下每个层次检索的文档数量
            
        Returns:
            str: 格式化后的上下文字符串
        """
        # 获取内容列表
        results = self.search(
            collection_name=collection_name,
            query=query,
            k=k,
            metadata_filter=metadata_filter,
            search_type=search_type,
            d=d,
            top_k=top_k
        )
        
        # 如果没有检索到内容，返回提示信息
        if not results:
            return "未检索到相关内容。"

        # 格式化每个文档片段
        formatted_chunks = []
        for i, result in enumerate(results, 1):
            # 清理文本（移除多余的空白字符）
            cleaned_content = " ".join(result["content"].split())
            # 添加编号和格式化
            formatted_chunk = f"[级别为{result['metadata']['level']}的文档片段 {i}]\n{cleaned_content}"
            formatted_chunks.append(formatted_chunk)
        
        # 使用分隔线连接所有文档片段
        formatted_context = "\n\n---\n\n".join(formatted_chunks)
        
        return formatted_context
        
        

