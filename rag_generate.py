
# 这个rag_retrieve 指向了 rag_retrieve.py 是我们多文件使用的一种导入方式
from rag_retrieve import query_similar_text  # 导入检索模块，负责从数据库中获取最相似的文本块
from zhipuai import ZhipuAI  # 用于调用智谱AI生成回答
from dotenv import load_dotenv  # 加载 .env 文件中的环境变量
import os  # 访问环境变量

# ======================
# 加载环境变量
# ======================
load_dotenv()  # 从 .env 文件中加载配置信息

# 获取 API 密钥和数据库路径
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")  # 从环境变量中读取智谱AI的API密钥
DATABASE_PATH = os.getenv("DATABASE_PATH")  # 从环境变量中读取数据库文件路径

# ======================
# 调用智谱AI生成回答
# ======================
def generate_answer(query: str, context: str, api_key: str) -> str:
    """
    使用智谱AI生成回答
    :param query: 用户的问题（自然语言文本）。
    :param context: 从数据库检索到的上下文文本。
    :param api_key: 智谱AI的API密钥。
    :return: 智谱AI生成的回答。
    """
    # 初始化智谱AI客户端
    client = ZhipuAI(api_key=ZHIPU_API_KEY)  # 使用传入的API密钥初始化客户端

    try:
        # 调用智谱AI生成接口
        response = client.chat.completions.create(
            model="glm-4-flash",  # 指定使用的大模型（此处使用 `glm-4-flash`）
            messages=[
                # 系统角色定义：设定AI助手的行为方式
                {"role": "system", "content": "你是一个知识库问答助手，请参考知识库的内容，回答用户问题。"},
                # 使用 `format` 格式化，将上下文插入到系统消息中
                {"role": "system", "content": "知识库内容{context}".format(context=context)},
                # 用户消息：使用 `f-string` 格式化，将用户问题插入消息内容
                {"role": "user", "content": f"{query}"}
            ]
        )
        # 返回生成的回答文本（从返回数据中提取回答内容）
        return response.choices[0].message.content
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
    :param api_key: 智谱AI的API密钥。
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
        api_key=ZHIPU_API_KEY,  # 智谱AI API 密钥
        top_k=3  # 从数据库中检索前3个相关文本块
    )

    # 打印最终回答
    print("最终回答：")
    print(final_answer)
