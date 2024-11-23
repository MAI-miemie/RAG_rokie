简易RAG使用说明文件，适用于 Python 3.12：

---

# RAG 系统使用说明

## 项目简介

本项目是一个基于检索增强生成 (RAG, Retrieval-Augmented Generation) 技术的简单实现。通过从知识库中检索相关内容并结合生成模型（智谱AI大模型）回答用户问题。

## 文件结构

```
project/
│
├── .env                 # 环境变量文件，存储 API 密钥和数据库路径
├── beiying.txt          # 示例文本文件，用于存储操作
├── rag_storage.py       # 数据存储模块：用于读取文档并生成向量存入数据库
├── rag_retrieve.py      # 数据检索模块：用于从数据库检索最相关内容
├── rag_generate.py      # 数据生成模块：用于生成最终回答
├── requirements.txt     # 项目依赖文件
```

---

## 环境要求

- **Python 版本**：3.12
- **依赖库**：
  项目依赖库已列在 `requirements.txt` 文件中，可以通过以下命令安装：

```bash
pip install -r requirements.txt
```

如果下载太慢，使用清华源加速下载

``` bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

`requirements.txt` 文件示例如下：
```plaintext
zhipuai
python-dotenv
numpy
```

---

## 使用步骤

### 1. 准备工作

#### **设置 `.env` 文件**
在项目根目录下创建 `.env` 文件，添加以下内容并替换为您的信息：
```plaintext
ZHIPU_API_KEY=your_zhipu_api_key_here
DATABASE_PATH=vector_database.db
```

- `ZHIPU_API_KEY`：智谱AI的API密钥。
- `DATABASE_PATH`：数据库文件路径，默认名为 `vector_database.db`。

#### **准备文档文件**
将待处理的文本文件（例如 `beiying.txt`）放置在项目目录中。

---

### 2. 存储阶段

运行 `rag_storage.py` 以读取文档内容，生成向量并存储到 SQLite 数据库：

```bash
python rag_storage.py
```

- 默认会读取 `.env` 中的 `DATABASE_PATH` 和文本文件路径，生成向量。
- 示例文件 `beiying.txt` 将被切分成小段并生成向量存储。

---

### 3. 检索阶段

运行 `rag_retrieve.py`，根据用户查询从数据库中检索最相关的文本块：

```bash
python rag_retrieve.py
```

示例：
```python
query_similar_text("问题内容", DATABASE_PATH, ZHIPU_API_KEY, top_k=3)
```

- **功能**：从数据库中获取前 `top_k` 个与查询最相关的文本块。

---

### 4. 生成功能

运行 `rag_generate.py`，结合检索结果生成最终回答：

```bash
python rag_generate.py
```

生成流程：
1. 从数据库中检索与查询相关的文本。
2. 调用智谱AI生成接口生成回答。

示例代码调用：
```python
retrieve_and_generate(
    query="用户问题",
    db_path="vector_database.db",
    api_key="your_zhipu_api_key_here",
    top_k=3
)
```

---

## 示例运行流程

1. **存储文档向量**：
   ```bash
   python rag_storage.py
   ```
   输出：
   ```plaintext
   文档切分为 7 个块。
   所有文本块已成功存入数据库。
   ```

2. **检索文档内容**：
   ```bash
   python rag_retrieve.py
   ```
   示例用户查询：
   ```plaintext
   谁给谁买了橘子？
   ```
   输出检索结果：
   ``` plaintext
    文件名: beiying.txt, 相似度: 0.3980, 文本块: 的迂;他们只认得钱，托他们直是白托!而且我这样大 年纪的人，难道还不能料理自己么?唉，我现在想想，那时真是太聪明 了!   我说道，“
    爸爸，你走吧。”他望车外看了看，说，“我买几个橘子 去。你就在此地，不要走动。”我看那边月台的栅栏外有几个卖东西的等 着顾客。走到那边月台，须穿过铁道，须跳下去又爬上去。父亲
    是一个胖 子，走过去自然要费事些。我本来要去的，他不肯，只好让他去。我看见 他戴着黑布小帽，
    文件名: beiying.txt, 相似度: 0.3740, 文本块: 他。他和我走到车上，将橘子一股脑儿放在我 的皮大衣上。于是扑扑衣上的泥土，心里很轻松似的，过一会说，“我走 了;到那边来信!”我望
    着他走出去。他走了几步，回过头看见我， 说，“进去吧，里边没人。”等他的背影混入来来往往的人里，再找不着 了，我便进来坐下，我的眼泪又来了。   近几年来，父亲和我都是东奔西走
    ，家中光景是一日不如一日。他少 年出外谋生，独力支持，做了许多大事。那知老境却如此颓唐!他触目
    文件名: beiying.txt, 相似度: 0.3651, 文本块: 穿着黑布大马褂，深青布棉袍，蹒跚地走到铁道边，慢 慢探身下去，尚不大难。可是他穿过铁道，要爬上那边月台，就不容易 了。他用两手 攀着上面，两脚再向上缩;他肥胖的身子向左微倾，显出努 力的样子。这时我看见他的背影，我的泪很快地流下来了。我赶紧拭干了 泪，怕他看见，也怕别人看见。我再向外看时，他已抱了朱 红的橘子望回 走了。过铁道时，他先将橘子散放在地上，自己慢慢爬下，再抱起橘子 走。到这边时，我赶紧去搀

```

3. **生成最终回答**：
   ```bash
   python rag_generate.py
   ```
   输出：
   ```plaintext
   最终回答：
根据上文描述，是父亲给“我”买了橘子。文中提到父亲看到月台栅栏外有卖橘子的，便决定去买橘子，并且将橘子放在“我”的皮大衣上。这显示了父亲对“我”的关心和照顾。
   ```

---

## 说明

- **文件扩展**：可根据需求修改 `rag_storage.py` 和 `rag_retrieve.py` 以支持不同文档格式（如 PDF、Word）。
- **测试使用**：可修改每个文件 `if __name__ == "__main__":` 来做测试。
- **效果拓展**：该代码只做了最简易的rag，效果达不到高水准。 

---

## 贡献者

- **开发者**：chengyibing@pami-ai.com

---