# ============================================
# INSTALL (Windows / Colab)
# pip install langchain langchain-groq langchain-huggingface langchain-community langgraph python-dotenv pinecone-client sentence-transformers
# ============================================

import os
import time
import json
import re
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, BaseMessage

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from pinecone import Pinecone, ServerlessSpec
from typing import Annotated, TypedDict


# ============================================
# 0. LOAD ENV VARIABLES
# ============================================
load_dotenv()


# ============================================
# 1. TEXT SANITIZER (CRITICAL FIX)
# ============================================
def clean_text(t: str):
    """Removes invalid UTF-8, surrogate pairs, broken emojis, null bytes."""
    if not isinstance(t, str):
        return ""
    t = t.replace("\x00", "")                                  # Remove null bytes
    t = t.encode("utf-8", "ignore").decode("utf-8")            # Drop invalid emojis
    t = re.sub(r"[\ud800-\udfff]", "", t)                      # Remove surrogate pairs
    return t.strip()


# ============================================
# 2. INIT GROQ LLM
# ============================================
llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)


# ============================================
# 3. EMBEDDINGS (HF)
# ============================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)


# ============================================
# 4. PINECONE INIT
# ============================================
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "linkedin-posts-rag"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

print(f"Index '{index_name}' is ready!")


# ============================================
# 5. LOAD DATA
# ============================================
with open("enriched_posts.json", "r", encoding="utf-8") as f:
    posts_data = json.load(f)


# ============================================
# 6. DOCUMENT CREATION (WITH SANITIZATION)
# ============================================
documents = []

for i, post in enumerate(posts_data):

    post_text = clean_text(post.get("text", ""))    # <-- FIX APPLIED

    if not post_text:
        continue

    doc = Document(
        page_content=post_text,
        metadata={
            "post_id": i,
            "engagement": int(post.get("engagement", 0)),
            "language": post.get("language", "Unknown"),
            "tags": post.get("tags", []),
            "tone": post.get("tone", "Neutral"),
            "line_count": int(post.get("line_count", 0)),
            "preview": post_text[:200]
        }
    )

    documents.append(doc)

print("Total posts loaded:", len(documents))


# ============================================
# 7. CHUNKING
# ============================================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)
chunks = [c for c in chunks if c.page_content.strip()]   # safety filter

print("Total chunks created:", len(chunks))


# ============================================
# 8. UPLOAD TO PINECONE
# ============================================
print("Uploading chunks to Pinecone...")

batch_size = 20
vector_store = None

for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    print(f"Batch {i//batch_size + 1}")

    if vector_store is None:
        vector_store = PineconeVectorStore.from_documents(
            documents=batch,
            embedding=embeddings,
            index_name=index_name
        )
    else:
        vector_store.add_documents(batch)

print("Upload complete!")


# ============================================
# 9. RETRIEVER
# ============================================
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)


# ============================================
# 10. TOOL: search_posts
# ============================================
@tool
def search_posts(query: str):
    """Semantic search across all LinkedIn posts."""
    results = retriever.invoke(query)
    return {
        "query": query,
        "results": [doc.page_content for doc in results],
        "metadata": [doc.metadata for doc in results],
        "count": len(results)
    }


tools = [search_posts]
llm_with_tools = llm.bind_tools(tools)


# ============================================
# 11. LANGGRAPH CHATBOT
# ============================================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile()


# ============================================
# 12. TEST QUERY
# ============================================
print("="*60)
print("Testing RAG Query...")
print("="*60)

result = chatbot.invoke({
    "messages": [HumanMessage(content="Find humorous posts about LinkedIn")]
})

print(result["messages"][-1].content)

print("\nRAG system ready!")
