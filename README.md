# RAG 入门项目 - 基于本地模型的检索增强生成系统

## 项目简介

本项目是一个基于**检索增强生成（RAG, Retrieval-Augmented Generation）**技术的完整实现。通过使用本地嵌入模型将文档转换为向量，结合相似度检索和智能回答生成，构建了一个完全本地运行的问答系统。

### 主要特点

- **完全免费**：使用本地模型，无需API调用费用
- **本地运行**：所有处理都在本地完成，保护数据隐私
- **即开即用**：一键运行，无需复杂配置
- **中文支持**：完美支持中文文本处理和问答
- **模块化设计**：清晰的代码结构，易于理解和扩展

## 项目结构

```
llm-RAGStarter/
├── .env                     # 环境变量配置文件
├── .gitignore              # Git忽略文件
├── beiying.txt             # 示例文档（朱自清《背影》）
├── rag_storage.py          # 数据存储模块：文档向量化存储
├── rag_retrieve.py         # 数据检索模块：相似度检索
├── rag_generate.py         # 答案生成模块：智能回答生成
├── requirements.txt        # 项目依赖
├── vector_database.db      # SQLite向量数据库
└── README.md              # 项目说明文档
```

## 技术栈

- **Python 3.12+**
- **Transformers**：Hugging Face本地模型库
- **PyTorch**：深度学习框架
- **SQLite**：轻量级向量数据库
- **NumPy**：数值计算和相似度计算
- **python-dotenv**：环境变量管理

## 安装依赖

### 1. 克隆项目
```bash
git clone https://github.com/MAI-miemie/RAG_rokie.git
cd llm-RAGStarter
```

### 2. 安装Python依赖
```bash
pip install -r requirements.txt
```

### 3. 配置环境变量
创建 `.env` 文件：
```env
DATABASE_PATH=vector_database.db
```

## 快速开始

### 步骤1：存储文档向量
```bash
python rag_storage.py
```
**输出示例：**
```
正在加载嵌入模型...
模型加载完成！
文档切分为 7 个块。
所有文本块已成功存入数据库。
```

### 步骤2：测试检索功能
```bash
python rag_retrieve.py
```
**输出示例：**
```
正在加载嵌入模型...
模型加载完成！
查询结果：
文件名: beiying.txt, 相似度: 0.7830, 文本块: 我说道，"爸爸，你走吧。"他望车外看了看，说，"我买几个橘子去。你就在此地，不要走动。"...
```

### 步骤3：完整RAG问答
```bash
python rag_generate.py
```
**输出示例：**
```
正在加载嵌入模型...
模型加载完成！
最终回答：
根据文本内容，是父亲给'我'买了橘子。文中提到父亲看到月台栅栏外有卖橘子的，便决定去买橘子，并且将橘子放在'我'的皮大衣上。这显示了父亲对'我'的关心和照顾。
```

## 核心模块说明

### 1. 存储模块 (`rag_storage.py`)
- **功能**：将文本文档转换为向量并存储到数据库
- **模型**：`sentence-transformers/all-MiniLM-L6-v2`
- **处理流程**：
  1. 读取文档内容
  2. 按200字符切分文本块
  3. 使用本地模型生成嵌入向量
  4. 存储到SQLite数据库

### 2. 检索模块 (`rag_retrieve.py`)
- **功能**：根据用户查询检索最相关的文本块
- **算法**：余弦相似度计算
- **返回**：按相似度排序的前K个文本块

### 3. 生成模块 (`rag_generate.py`)
- **功能**：基于检索结果生成智能回答
- **策略**：基于规则的答案生成
- **特点**：针对不同问题类型提供专门回答

## 性能指标

- **嵌入模型**：384维向量
- **文本切分**：200字符/块
- **检索精度**：相似度阈值 > 0.7
- **响应时间**：首次加载模型约10-15秒，后续查询 < 1秒

## 使用示例

### 示例1：基础问答
```python
from rag_generate import retrieve_and_generate

# 查询问题
query = "谁给谁买了橘子？"
answer = retrieve_and_generate(query, "vector_database.db", "", top_k=3)
print(answer)
```

### 示例2：自定义文档
```python
# 1. 将您的文档内容保存为 .txt 文件
# 2. 修改 rag_storage.py 中的文档路径
# 3. 重新运行存储流程
```

## 常见问题

### Q: 首次运行很慢怎么办？
A: 首次运行需要下载模型文件（约90MB），这是正常现象。后续运行会直接使用缓存的模型。

### Q: 如何添加新的文档？
A: 将新文档保存为 `.txt` 文件，修改 `rag_storage.py` 中的 `document_path` 变量，然后重新运行存储流程。

### Q: 如何提高检索精度？
A: 可以调整以下参数：
- 文本切分大小（`chunk_size`）
- 检索返回数量（`top_k`）
- 相似度阈值

### Q: 支持其他语言吗？
A: 当前模型主要针对英文优化，对中文也有良好支持。如需更好的中文支持，可以替换为中文专用模型。

## 自定义配置

### 修改文本切分大小
```python
# 在 rag_storage.py 中修改
chunks = split_text(text, chunk_size=300)  # 改为300字符
```

### 调整检索数量
```python
# 在 rag_generate.py 中修改
top_k=5  # 返回前5个最相关结果
```

### 更换嵌入模型
```python
# 在 rag_storage.py 和 rag_retrieve.py 中修改
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

## 扩展建议

1. **添加更多文档格式支持**：PDF、Word、Markdown等
2. **集成大语言模型**：使用GPT、Claude等生成更智能的回答
3. **Web界面**：开发Web应用提供更友好的交互
4. **多语言支持**：添加更多语言的嵌入模型
5. **性能优化**：使用向量数据库（如Chroma、Pinecone）提升检索效率

## 贡献指南

欢迎提交Issue和Pull Request来改进这个项目！

### 开发环境设置
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装开发依赖
pip install -r requirements.txt
```

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 贡献者

- **开发者**：MieMie
- **维护者**：MieMie

## 致谢

- [Hugging Face](https://huggingface.co/) - 提供优秀的预训练模型
- [Sentence Transformers](https://www.sbert.net/) - 强大的句子嵌入库
- [PyTorch](https://pytorch.org/) - 深度学习框架

---

**如果这个项目对您有帮助，请给个Star支持一下！**
