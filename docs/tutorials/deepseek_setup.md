# 使用 DeepSeek API 配置 PaperQA

PaperQA 底层使用 [LiteLLM](https://docs.litellm.ai) 统一对接 100+ 模型提供商，DeepSeek 是 LiteLLM 原生支持的提供商之一。本教程介绍三种配置方式。

---

## 方式一：环境变量（最简单）

设置环境变量 `DEEPSEEK_API_KEY`，然后在代码中直接指定模型名称。

```python
import os
from paperqa import Settings, ask

os.environ["DEEPSEEK_API_KEY"] = "sk-your-deepseek-api-key"

settings = Settings(
    llm="deepseek/deepseek-chat",           # DeepSeek-V3
    summary_llm="deepseek/deepseek-chat",
    embedding="text-embedding-3-small",      # 嵌入模型仍用 OpenAI
)

response = ask("Your scientific question?", settings=settings)
print(response.answer)
```

可用模型名：

| 模型 | LiteLLM 名称 |
|---|---|
| DeepSeek-V3 | `deepseek/deepseek-chat` |
| DeepSeek-R1 | `deepseek/deepseek-reasoner` |

**注意**：嵌入模型推荐仍使用 OpenAI 的 `text-embedding-3-small`（需设置 `OPENAI_API_KEY`），或使用本地 Sentence Transformer / Ollama 嵌入模型（见下文）。

---

## 方式二：自定义 API 基础地址（`llm_config`）

如果需要指定自定义的 API 基础地址（例如使用代理或自建推理服务），使用 `llm_config` / `summary_llm_config` 参数传入 LiteLLM 路由器配置：

```python
from paperqa import Settings, ask

settings = Settings(
    llm="deepseek-chat-custom",
    llm_config={
        "model_list": [
            {
                "model_name": "deepseek-chat-custom",
                "litellm_params": {
                    "model": "openai/deepseek-chat",   # "openai/" 前缀表示兼容 OpenAI 的 API
                    "api_base": "https://api.deepseek.com/v1",
                    "api_key": "sk-your-deepseek-api-key",
                    "temperature": 0.0,
                },
            }
        ]
    },
    summary_llm="deepseek-chat-custom",
    summary_llm_config={...},  # 同上
    embedding="text-embedding-3-small",
)
```

也可以用 `"deepseek/"` 前缀让 LiteLLM 使用原生 DeepSeek 集成：

```python
litellm_params = {
    "model": "deepseek/deepseek-chat",
    "api_base": "https://api.deepseek.com/v1",
    "api_key": "sk-your-deepseek-api-key",
}
```

---

## 方式三：通过 JSON 配置文件

将配置保存到 `~/.pqa/settings/deepseek.json`：

```json
{
  "llm": "deepseek/deepseek-chat",
  "llm_config": {
    "model_list": [
      {
        "model_name": "deepseek/deepseek-chat",
        "litellm_params": {
          "model": "deepseek/deepseek-chat",
          "api_base": "https://api.deepseek.com/v1"
        }
      }
    ]
  },
  "summary_llm": "deepseek/deepseek-chat",
  "summary_llm_config": {
    "model_list": [
      {
        "model_name": "deepseek/deepseek-chat",
        "litellm_params": {
          "model": "deepseek/deepseek-chat",
          "api_base": "https://api.deepseek.com/v1"
        }
      }
    ]
  },
  "embedding": "text-embedding-3-small"
}
```

然后在代码中加载：

```python
from paperqa import Settings

settings = Settings.from_name("deepseek")
```

或者通过 CLI 使用：

```bash
export DEEPSEEK_API_KEY=sk-...
pqa --config deepseek ask "Your question?"
```

---

## 使用 DeepSeek-R1（推理模型）

```python
import os
from paperqa import Settings, ask

os.environ["DEEPSEEK_API_KEY"] = "sk-your-deepseek-api-key"

settings = Settings(
    llm="deepseek/deepseek-reasoner",
    summary_llm="deepseek/deepseek-reasoner",
    embedding="text-embedding-3-small",
)
```

---

## 完整示例：PDF 问答

```python
import os
from paperqa import Settings, ask

os.environ["DEEPSEEK_API_KEY"] = "sk-your-deepseek-api-key"
# 如果使用 OpenAI 嵌入，也需要设置：
os.environ["OPENAI_API_KEY"] = "sk-your-openai-key"

settings = Settings(
    llm="deepseek/deepseek-chat",
    summary_llm="deepseek/deepseek-chat",
    embedding="text-embedding-3-small",
    temperature=0.1,
)

response = ask(
    "What are the key contributions of this paper?",
    settings=settings,
    paper_paths=["./my_papers/paper.pdf"],
)
print(f"Answer: {response.answer}")
print(f"Used {len(response.contexts)} passages")
```

---

## 使用本地嵌入模型（无需 OpenAI）

如果想完全脱离 OpenAI，有两种本地嵌入方案。

### 方案一：Sentence Transformers

```python
from paperqa import Settings

settings = Settings(
    llm="deepseek/deepseek-chat",
    summary_llm="deepseek/deepseek-chat",
    embedding="BAAI/bge-small-en-v1.5",
)
```

需要安装 `pip install paper-qa[local]`。

### 方案二：本地 Ollama 嵌入

[Ollama](https://ollama.ai) 支持多种嵌入模型，可以完全本地运行。

**第一步：拉取嵌入模型**

```bash
ollama pull nomic-embed-text
# 或 mxbai-embed-large（更大，更准确）
ollama pull mxbai-embed-large
```

**第二步：配置 PaperQA 使用 Ollama 嵌入**

Ollama 嵌入通过 `embedding_config` 传入，需指定 `api_base`：

```python
from paperqa import Settings, ask

settings = Settings(
    llm="deepseek/deepseek-chat",
    summary_llm="deepseek/deepseek-chat",
    embedding="ollama/nomic-embed-text",
    embedding_config={
        "model_list": [
            {
                "model_name": "ollama/nomic-embed-text",
                "litellm_params": {
                    "model": "ollama/nomic-embed-text",
                    "api_base": "http://localhost:11434",
                },
            }
        ]
    },
)

response = ask("Your question?", settings=settings)
```

**完整示例：DeepSeek LLM + Ollama 嵌入（完全本地嵌入，无需任何云端 API）**

```python
import os
from paperqa import Settings, ask

os.environ["DEEPSEEK_API_KEY"] = "sk-your-deepseek-api-key"

settings = Settings(
    llm="deepseek/deepseek-chat",
    summary_llm="deepseek/deepseek-chat",
    embedding="ollama/nomic-embed-text",
    embedding_config={
        "model_list": [
            {
                "model_name": "ollama/nomic-embed-text",
                "litellm_params": {
                    "model": "ollama/nomic-embed-text",
                    "api_base": "http://localhost:11434",
                },
            }
        ]
    },
)

response = ask(
    "What are the key contributions of this paper?",
    settings=settings,
    paper_paths=["./my_papers/paper.pdf"],
)
print(f"Answer: {response.answer}")
```

**常见 Ollama 嵌入模型**

| 模型 | 参数 | 维度 |
|---|---|---|
| `nomic-embed-text` | 137M | 768 |
| `mxbai-embed-large` | 334M | 1024 |
| `all-minilm` | 33M | 384 |

---

## 参考

- [LiteLLM DeepSeek 文档](https://docs.litellm.ai/docs/providers/deepseek)
- [PaperQA 配置文档](https://github.com/Future-House/paper-qa)
- [DeepSeek API 文档](https://platform.deepseek.com/api-docs)
