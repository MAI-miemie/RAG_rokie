import sqlite3  # 用于操作SQLite数据库
import pickle  # 用于反序列化数据
from typing import List, Tuple
from zhipuai import ZhipuAI  # 用于调用智谱AI生成Embedding
from dotenv import load_dotenv  # 用于加载 .env 文件中的环境变量
import os  # 用于读取环境变量
import numpy as np  # 用于计算余弦相似度，numpy是一个非常经典的计算库
# ps： 你可以手动实现向量的计算（不推荐），numpy只是做了封装

# ======================
# 加载环境变量
# ======================

# 这个文件中我们没有指定路径，他会自动从路径 .env 导入
load_dotenv()  # 从 .env 文件加载环境变量

# 获取 API 密钥和数据库路径
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")  # 从 .env 文件获取智谱AI API 密钥
DATABASE_PATH = os.getenv("DATABASE_PATH")  # 从 .env 文件获取数据库路径

# ======================
# 初始化智谱AI接口
# ======================
def initialize_zhipu(api_key: str):
    """
    设置智谱AI的API密钥。
    :param api_key: 智谱AI的API密钥。
    """
    #同存储

# ======================
# 调用智谱AI生成Embedding
# ======================
def generate_embedding(text: str) -> List[float]:
    """
    使用智谱AI生成文本的向量（Embedding）。
    :param text: 输入的文本字符串。
    :return: 文本的Embedding（浮点数列表）。
    """
    try:
       # 初始化智谱AI客户端,注意初始化客户端需要传入apikey
        client = ZhipuAI(api_key=ZHIPU_API_KEY)
        # 调用智谱AI的Embedding接口，将文本传入
        response = client.embeddings.create(
            model="embedding-3", #填写需要调用的模型编码
            input= text, # 传入需要生成embedding的文本                            
                )
        # 返回生成的Embedding向量,因为我们只传入了一个，所以只返回第0条数据
        return response.data[0].embedding
    except Exception as e:
        print(f"生成Embedding时出错: {e}")
        return []

# ======================
# 从数据库中检索数据
# ======================
def fetch_embeddings(db_path: str) -> List[Tuple[str, str, List[float]]]:
    """
    从数据库中提取存储的文本块及其Embedding。
    :param db_path: 数据库文件路径。
    :return: 返回一个包含(file_name, chunk, embedding)的列表。
    """
    conn = sqlite3.connect(db_path)  # 连接到数据库
    query = "SELECT file_name, chunk, embedding FROM embeddings"  # 查询所有存储的文本块及Embedding
    results = []
    
    for row in conn.execute(query):
        file_name, chunk, serialized_embedding = row
        # 反序列化Embedding
        embedding = pickle.loads(serialized_embedding)
        results.append((file_name, chunk, embedding))
    
    conn.close()  # 关闭数据库连接
    return results

# ======================
# 计算余弦相似度
# ======================
def calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    计算两个向量之间的余弦相似度。
    :param vec1: 第一个向量。
    :param vec2: 第二个向量。
    :return: 余弦相似度（介于-1到1之间）。
    """
    # 如果你对为什么余弦相似可以代表语义相近感兴趣，可以查看相关文章，这里不做解释
    vec1 = np.array(vec1)  # 转为NumPy数组
    vec2 = np.array(vec2)  # 转为NumPy数组
    dot_product = np.dot(vec1, vec2)  # 点积
    norm1 = np.linalg.norm(vec1)  # vec1的模
    norm2 = np.linalg.norm(vec2)  # vec2的模
    return dot_product / (norm1 * norm2)  # 余弦相似度公式

# ======================
# 查询最相似的文本块
# ======================
def query_similar_text(query: str, db_path: str, api_key: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
    """
    根据查询文本，从数据库中检索最相似的文本块。
    :param query: 用户的查询文本。
    :param db_path: 数据库文件路径。
    :param api_key: 智谱AI的API密钥。
    :param top_k: 返回最相似的前K个文本块。
    :return: 返回一个包含(file_name, chunk, similarity)的列表，按相似度降序排列。
    """
    # 1. 设置智谱AI API密钥
    initialize_zhipu(api_key)

    # 2. 对查询文本生成Embedding
    query_embedding = generate_embedding(query)
    if not query_embedding:
        print("生成查询文本的Embedding失败！")
        return []

    # 3. 从数据库中获取存储的文本块及其Embedding
    stored_data = fetch_embeddings(db_path)

    # 4. 计算相似度
    similarities = []
    for file_name, chunk, embedding in stored_data:
        similarity = calculate_cosine_similarity(query_embedding, embedding)
        similarities.append((file_name, chunk, similarity))

    # 5. 按相似度降序排列并返回前K个结果
    similarities = sorted(similarities, key=lambda x: x[2], reverse=True)
    return similarities[:top_k]

# ======================
# 示例调用
# ======================
if __name__ == "__main__":
    # 示例查询
    user_query = "谁给谁买了橘子"  # 替换为你的查询
    top_k_results = query_similar_text(user_query, DATABASE_PATH, ZHIPU_API_KEY, top_k=3)

    # 打印结果
    print("查询结果：")
    for file_name, chunk, similarity in top_k_results:
        print(f"文件名: {file_name}, 相似度: {similarity:.4f}, 文本块: {chunk}")
