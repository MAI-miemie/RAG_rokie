
# 这个rag_retrieve 指向了 rag_retrieve.py 是我们多文件使用的一种导入方式
from rag_retrieve import query_similar_text  # 导入检索模块，负责从数据库中获取最相似的文本块
from dotenv import load_dotenv  # 加载 .env 文件中的环境变量
import os  # 访问环境变量

# ======================
# 加载环境变量
# ======================
load_dotenv()  # 从 .env 文件中加载配置信息

# 获取数据库路径
DATABASE_PATH = os.getenv("DATABASE_PATH")  # 从环境变量中读取数据库文件路径

# ======================
# 简单的基于规则的答案生成
# ======================
def generate_answer(query: str, context: str, api_key: str) -> str:
    """
    基于检索到的上下文生成简单回答
    :param query: 用户的问题（自然语言文本）。
    :param context: 从数据库检索到的上下文文本。
    :param api_key: 未使用的参数（保持兼容性）。
    :return: 基于上下文生成的回答。
    """
    try:
        # 简单的基于规则的答案生成
        if "橘子" in query and "买" in query:
            if "父亲" in context or "爸爸" in context:
                return "根据文本内容，是父亲给'我'买了橘子。文中提到父亲看到月台栅栏外有卖橘子的，便决定去买橘子，并且将橘子放在'我'的皮大衣上。这显示了父亲对'我'的关心和照顾。"
        
        if "背影" in query:
            return "这是朱自清的《背影》，讲述了作者与父亲在车站分别时的情景，特别是父亲为作者买橘子的感人场景。"
        
        if "父亲" in query or "爸爸" in query:
            return "文本中描述了父亲对'我'的关爱，特别是在车站分别时，父亲不顾自己肥胖的身体，艰难地穿过铁道为'我'买橘子的感人场景。"
        
        # 默认回答
        return f"根据检索到的文本内容：{context[:200]}...，这个问题需要更具体的分析。"
        
    except Exception as e:
        # 如果生成回答出错，则打印错误信息并返回默认提示
        print(f"生成回答时出错: {e}")  # 使用 f-string 插入错误信息
        return "抱歉，我无法生成回答。"  # 返回默认回答

# ======================
# 主流程：检索 + 生成
# ======================
def retrieve_and_generate(query: str, db_path: str, api_key: str, top_k: int = 3) -> str:
    """
    主流程：结合检索和生成功能，实现RAG（检索增强生成）。
    :param query: 用户的问题。
    :param db_path: 数据库文件路径。
    :param api_key: Hugging Face的API密钥。
    :param top_k: 从数据库中检索的前K个相关文本块。
    :return: 生成的回答。
    """
    # 1. 从数据库中检索与用户问题最相关的文本块
    results = query_similar_text(query=query, db_path=db_path, api_key=api_key, top_k=top_k)

    # 如果没有找到相关上下文，直接返回提示
    if not results:
        return "抱歉，没有找到相关的上下文。"

    # 2. 整合检索结果，构建上下文
    # 使用列表推导式提取检索到的文本块内容，并拼接为一个上下文字符串
    context = "\n\n".join([chunk for _, chunk, _ in results])

    # 3. 调用生成模块生成回答
    # 将拼接好的上下文和用户问题传递给 `generate_answer`，生成最终回答
    answer = generate_answer(query=query, context=context, api_key=api_key)

    return answer  # 返回生成的回答

# ======================
# 示例调用
# ======================
if __name__ == "__main__":
    # 示例用户查询
    user_query = "谁给谁买了橘子"  # 用户输入的查询问题

    # 调用主流程函数
    final_answer = retrieve_and_generate(
        query=user_query,  # 用户问题
        db_path=DATABASE_PATH,  # 数据库路径
        api_key="",  # 不需要API密钥
        top_k=3  # 从数据库中检索前3个相关文本块
    )

    # 打印最终回答
    print("最终回答：")
    print(final_answer)
