import sys
from pathlib import Path

# 获取项目根目录
project_root = Path(__file__).parent.parent.absolute()
# 将项目根目录添加到 sys.path
sys.path.append(str(project_root))

# 导入模块
from vdb_managers.drop.chroma_manager_2 import ChromaManager

# 创建ChromaVectorStore实例 - 初始化的时候可以自定义嵌入模型，collection_name和persist_directory
chroma_store = ChromaManager() # 自定义的chroma类

# 上传PDF文档
chroma_store.upload_pdf_file("files/论文 - GraphRAG.pdf")