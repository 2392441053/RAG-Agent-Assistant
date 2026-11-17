import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

st.set_page_config(page_title="智能学习助手", page_icon="🤖")
st.title("🤖 个人智能学习助手")

# 侧边栏配置
with st.sidebar:
    api_key = st.text_input("请输入 API Key", type="password")
    uploaded_file = st.file_uploader("上传你的学习资料 (PDF)", type="pdf")

if api_key and uploaded_file:
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = "https://open.bigmodel.cn/api/paas/v4/"

    # 保存上传的文件到本地
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())


    # 初始化 RAG (用 st.cache_resource 避免重复加载)
    @st.cache_resource
    def init_rag():
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings(model="embedding-2")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        return vectorstore


    try:
        vectorstore = init_rag()
        llm = ChatOpenAI(model="glm-3-turbo", temperature=0.1)

        # 初始化聊天记录
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 显示历史聊天记录
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 用户输入
        if prompt := st.chat_input("针对文档提问..."):
            # 显示用户的问题
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # AI 回答
            with st.chat_message("assistant"):
                with st.spinner("思考中..."):
                    # 1. 检索相关文档
                    docs = vectorstore.similarity_search(prompt, k=3)
                    context = "\n".join([doc.page_content for doc in docs])

                    # 2. 构建历史记录文本
                    history_text = ""
                    for msg in st.session_state.messages[-5:-1]:  # 取最近两轮对话
                        history_text += f"{msg['role']}: {msg['content']}\n"

                    # 3. 拼装 Prompt 直接发给大模型
                    final_prompt = f"""你是一个智能学习助手。请根据以下参考资料和对话历史回答用户问题。
如果参考资料中没有答案，请直接说“根据提供的文档，我无法回答这个问题”，不要编造。

【参考资料】：
{context}

【历史对话】：
{history_text}

【用户问题】：
{prompt}
"""
                    response = llm.invoke(final_prompt)
                    st.markdown(response.content)

            # 保存 AI 的回答到记录
            st.session_state.messages.append({"role": "assistant", "content": response.content})

    except Exception as e:
        st.error(f"程序运行出错，请检查 API Key 或 PDF 文件。详细报错：{e}")