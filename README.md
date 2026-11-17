# 🤖 个人智能学习助手 (RAG Agent)

基于 LangChain 与大语言模型（LLM）构建的文档问答助手，支持上传 PDF 文件并进行多轮对话。

## 🌟 核心功能
- **RAG 检索增强生成**：精准基于上传的 PDF 内容回答问题，拒绝“幻觉”。
- **多轮对话记忆**：支持上下文连贯对话。
- **前端可视化**：基于 Streamlit 搭建，支持动态输入 API Key 和上传文档。

## 🛠️ 技术栈
- Python, LangChain, Streamlit, ChromaDB, 智谱 GLM API

## 🚀 如何运行
1. 克隆项目：`git clone <你的仓库链接>`
2. 安装依赖：`pip install langchain langchain-community langchain-openai streamlit pypdf chromadb`
3. 运行程序：`streamlit run app.py`
4. 在网页左侧输入你的大模型 API Key 即可使用。