import os
import sqlite3  # Python内置的SQLite数据库模块，用于创建和操作轻量级数据库
from typing import List  # 用于定义函数返回值的类型
import pickle  # 用于将复杂对象（如列表）序列化为二进制格式
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
# ======================
# 加载环境变量
# ======================

#注意这里的 .env 文件是指的存有环境变量数据的文件路径，可以是相对路径，也可以是绝对路径，他可以不叫.env 只要内容格式正确即可
load_dotenv(".env")  # 从 .env 文件中加载环境变量

# 获取数据库路径
DATABASE_PATH = os.getenv("DATABASE_PATH")  # 从 .env 文件获取数据库路径

# 初始化模型和分词器
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = None
model = None

# ======================
# 初始化本地模型
# ======================
def initialize_model():
    """
    初始化本地嵌入模型
    """
    global tokenizer, model
    try:
        print("正在加载嵌入模型...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        print("模型加载完成！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        raise e

# ======================
# 使用本地模型生成Embedding
# ======================
def generate_embedding(text: str) -> List[float]:
   """
   使用本地模型生成指定文本的向量（Embedding）。
   :param text: 输入的文本字符串。
   :return: 返回一个浮点数列表，表示文本的Embedding向量。
   """
   try:
       # 对文本进行分词
       inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
       
       # 生成嵌入向量
       with torch.no_grad():
           outputs = model(**inputs)
           # 使用[CLS]标记的嵌入作为句子嵌入
           embeddings = outputs.last_hidden_state[:, 0, :].numpy()
       
       # 返回嵌入向量的列表形式
       return embeddings[0].tolist()
   except Exception as e:
       # 如果生成失败，打印错误信息
       print(f"生成Embedding时出错: {e}")
       return []  # 返回一个空列表表示失败

# ======================
# 将文本按固定长度切分
# ======================
def split_text(text: str, chunk_size: int = 200) -> List[str]:
    """
    将输入文本按固定长度切分为多个小块。
    :param text: 输入的长文本字符串。
    :param chunk_size: 每块的字符长度，默认为200。
    :return: 返回一个字符串列表，每个元素是一个长度不超过chunk_size的文本块。
    """
    # 用replace将换行符替换为空格，方便连续处理
    text = text.replace("\n", " ")
    # 使用列表推导式，每chunk_size长度切一次
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# ======================
# 读取文档内容
# ======================
def read_document(file_path: str) -> str:
    """
    从指定路径读取文本文档内容。
    :param file_path: 文档的文件路径，支持TXT格式。
    :return: 返回文档的内容字符串。
    """
    # 打开文件，使用 UTF-8 编码读取所有内容
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()  # 将文件内容读取为字符串返回

# ======================
# 初始化数据库
# ======================
def initialize_database(db_path: str = "vector_database.db"):
    """
    创建一个SQLite数据库，并初始化存储表格。
    :param db_path: 数据库文件路径（默认文件名为"vector_database.db"）。
    """
    # 连接到数据库文件，如果文件不存在则自动创建
    conn = sqlite3.connect(db_path)
    # SQL语句：创建存储Embedding的表格
    query = """
    CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,  -- 自动递增的唯一标识
        file_name TEXT,                       -- 文档的文件名
        chunk TEXT,                           -- 文本块内容
        embedding BLOB                        -- 文本块的Embedding（以二进制形式存储）
    );
    """
    conn.execute(query)  # 执行SQL语句
    conn.commit()  # 提交更改到数据库
    conn.close()  # 关闭数据库连接

# ======================
# 插入Embedding到数据库
# ======================
def insert_embedding(db_path: str, file_name: str, chunk: str, embedding: List[float]):
    """
    将文本块及其Embedding插入到数据库中。
    :param db_path: 数据库文件路径。
    :param file_name: 文档文件名。
    :param chunk: 文本块内容。
    :param embedding: 文本块对应的Embedding（向量）。
    """
    # 连接到SQLite数据库
    conn = sqlite3.connect(db_path)
    # 插入数据的SQL语句
    query = "INSERT INTO embeddings (file_name, chunk, embedding) VALUES (?, ?, ?)"
    # 将浮点列表转换为二进制存储（SQLite不支持直接存储列表。
    # 注意我们传递的顺序，三个问号对应下面传递的三个值，这是python里sqlite的一种语法
    
    # 使用pickle将浮点数列表序列化为二进制数据
    serialized_embedding = pickle.dumps(embedding)
    conn.execute(query, (file_name, chunk, sqlite3.Binary(serialized_embedding )))
    conn.commit()  # 提交更改到数据库
    conn.close()  # 关闭数据库连接

# ======================
# 主流程：存储Embedding
# ======================
def store_embeddings(file_path: str, api_key: str, db_path: str = "vector_database.db"):
    """
    主流程：从文档读取文本，生成Embedding，并存储到数据库中。
    :param file_path: 文档路径。
    :param api_key: Hugging Face的API密钥。
    :param db_path: 数据库文件路径。
    """
    # 1. 初始化本地模型
    initialize_model()

    # 2. 初始化数据库
    initialize_database(db_path)

    # 3. 读取文档内容
    text = read_document(file_path)  # 将文档内容读取为字符串
    file_name = file_path.split("/")[-1]  # 提取文件名

    # 4. 切分文本
    chunks = split_text(text)  # 将长文本按固定长度切分
    print(f"文档切分为 {len(chunks)} 个块。")  # 打印切分后的块数

    # 5. 遍历切分的文本块，生成Embedding并存储
    for chunk in chunks:
        # 调用智谱AI生成当前文本块的Embedding
        embedding = generate_embedding(chunk)
        if embedding:  # 如果生成成功
            # 将文本块及其Embedding插入到数据库
            insert_embedding(db_path, file_name, chunk, embedding)

    print("所有文本块已成功存入数据库。")  # 提示存储完成

# ======================
# 示例调用
# ======================
# 这里对 if __name__ == "__main__"做解释  
# if __name__ == "__main__": 是 Python 中一个常见的结构，用于区分脚本是被直接运行还是被作为模块导入。
# 结构的含义
# __name__ 是 Python 的一个内置变量：
# 当 Python 脚本被直接运行时，__name__ 的值为 "__main__"。
# 当脚本被作为模块导入时，__name__ 的值为模块的名称（通常是文件名，不包含扩展名）。
# if __name__ == "__main__": 的作用：
# 它允许你控制某些代码仅在脚本被直接运行时执行，而在模块被导入时不会执行。
# 应用场景
# 区分直接运行和模块导入
# 当直接运行脚本时，if __name__ == "__main__": 下的代码会被执行。
# 当脚本被作为模块导入到其他代码中时，这部分代码不会被执行。
# 组织代码
# 将测试代码、脚本执行入口或其他需要运行的逻辑放在 if __name__ == "__main__": 块中，保持模块作为库导入时的整洁。
if __name__ == "__main__":
    # 替换为你的文档路径
    document_path = "beiying.txt"  # 替换为要处理的文档路径
    database_path = DATABASE_PATH  # 数据库存储路径
    # 调用主函数开始处理
    store_embeddings(document_path, "", database_path)
