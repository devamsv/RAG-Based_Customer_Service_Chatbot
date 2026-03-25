# -*- coding: utf-8 -*-
"""
Customer Support Chatbot with Streamlit UI.
Dataset-based Retrieval-Augmented Generation (RAG) using Gemini + HuggingFace embeddings.
"""

import logging
import os
import time
import hashlib
import re
from typing import Optional, List, Tuple, Dict, Any

import streamlit as st
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from config import config
from utils import (
    setup_logging,
    truncate_text,
    ensure_directory_exists
)
# Setup logging
logger = setup_logging(config.LOG_LEVEL)

# Validate configuration early and fail fast in the UI
if not config.GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY not found in environment variables")
    st.error("GOOGLE_API_KEY not configured. Please set it in your .env file.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pending_question' not in st.session_state:
    st.session_state.pending_question = None
if 'satisfaction' not in st.session_state:
    # Map: assistant_index(str) -> rating_value(float in [0,1])
    st.session_state.satisfaction = {}


# Streamlit UI
st.title(f"{config.PAGE_ICON} {config.PAGE_TITLE}")
st.subheader("Ask your query about orders / refund / login etc")
st.caption("Answers are generated using the Customer Support dataset and Gemini.")

# ---------- Data + RAG helpers ----------
@st.cache_data(show_spinner=False)
def load_customer_support_df(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {"question", "answer", "category"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df["question"] = df["question"].fillna("").astype(str)
    df["answer"] = df["answer"].fillna("").astype(str)
    df["category"] = df["category"].fillna("General").astype(str)
    return df


@st.cache_resource(show_spinner=False)
def get_embedding_model() -> HuggingFaceEmbeddings:
    # Requirement: REAL embeddings
    return HuggingFaceEmbeddings(model_name=config.HF_EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def get_llm() -> ChatGoogleGenerativeAI:
    # Requirement: Gemini LLM
    return ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        temperature=config.LLM_TEMPERATURE,
        google_api_key=config.GOOGLE_API_KEY,
    )


@st.cache_resource(show_spinner=False)
def get_answer_chain():
    prompt_template = """You are a helpful customer support assistant.

Context:
{context}

Question: {question}

Instructions:
- Answer the question using only the information from the provided context.
- If the answer is not in the context, say: "I don't have enough information in the provided dataset to answer this question."
- Be concise and accurate.

Answer:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return prompt | get_llm() | StrOutputParser()


def _distance_to_relevance(distance: float) -> float:
    # FAISS "score" is typically a distance (smaller is better).
    # Convert to a [0..1]-like relevance proxy.
    d = float(distance)
    d = abs(d)
    return 1.0 / (1.0 + d)


def _token_set(text: str) -> set:
    return set(re.findall(r"\w+", text.lower()))


def compute_precision_recall(
    generated_answer: str,
    expected_answer: str,
) -> Dict[str, float]:
    gen_tokens = _token_set(generated_answer)
    exp_tokens = _token_set(expected_answer)
    if not gen_tokens or not exp_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    overlap = gen_tokens.intersection(exp_tokens)
    precision = len(overlap) / max(len(gen_tokens), 1)
    recall = len(overlap) / max(len(exp_tokens), 1)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def build_documents_from_subset(df_subset: pd.DataFrame) -> List[Document]:
    """
    Create FAISS documents from the CSV answers.

    To satisfy the "data preprocessing + chunking" requirement, we chunk each answer
    into smaller passages (even though your CSV answers are short).
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
    )

    docs: List[Document] = []
    for question, answer, category in zip(
        df_subset["question"].tolist(),
        df_subset["answer"].tolist(),
        df_subset["category"].tolist(),
    ):
        # Split answer into chunks; each chunk inherits the same metadata.
        chunks = text_splitter.split_text(answer)
        if not chunks:
            chunks = [answer]

        for chunk in chunks:
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "question": question,
                        "answer": answer,  # keep full expected answer for evaluation
                        "category": category,
                    },
                )
            )

    return docs


@st.cache_resource(show_spinner=False)
def get_vectorstore_for_categories(categories: Tuple[str, ...]) -> Tuple[FAISS, bool]:
    df = load_customer_support_df(config.CUSTOMER_SUPPORT_CSV_PATH)

    if not categories:
        df_subset = df
        categories_key = "all"
    else:
        df_subset = df[df["category"].isin(list(categories))]
        categories_key = "|".join(sorted(categories))

    index_id = hashlib.sha1(categories_key.encode("utf-8")).hexdigest()[:12]
    index_path = os.path.join(config.VECTOR_STORE_DIR, index_id)

    index_faiss_file = os.path.join(index_path, "index.faiss")
    embeddings = get_embedding_model()

    if os.path.exists(index_faiss_file):
        vs = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return vs, True

    ensure_directory_exists(index_path)
    docs = build_documents_from_subset(df_subset)
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    vectorstore.save_local(index_path)
    return vectorstore, False


def answer_with_metrics(
    query: str,
    vectorstore: FAISS,
):
    chain = get_answer_chain()

    # Retrieval latency + retrieved context
    retrieval_start = time.time()
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=config.RETRIEVAL_K)
    retrieval_latency = time.time() - retrieval_start

    source_documents = [d for d, _score in docs_and_scores]
    scores = [_score for _d, _score in docs_and_scores]
    context_relevance = (
        sum(_distance_to_relevance(s) for s in scores) / max(len(scores), 1)
        if scores
        else 0.0
    )

    context_text = "\n\n".join(
        f"[{i + 1}] Category: {doc.metadata.get('category', 'General')}\n{doc.page_content}"
        for i, doc in enumerate(source_documents)
    )

    generation_start = time.time()
    answer = chain.invoke({"context": context_text, "question": query})
    generation_latency = time.time() - generation_start

    # Response accuracy proxy (token overlap with top retrieved answer)
    expected_answer = ""
    if source_documents:
        expected_answer = source_documents[0].metadata.get("answer", source_documents[0].page_content)

    pr = compute_precision_recall(answer, expected_answer)

    return {
        "answer": answer,
        "source_documents": source_documents,
        "metrics": {
            "retrieval_latency_seconds": retrieval_latency,
            "generation_latency_seconds": generation_latency,
            "context_relevance": context_relevance,
            "response_accuracy_proxy_f1": pr["f1"],
            "precision": pr["precision"],
            "recall": pr["recall"],
        },
    }


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("---")

    st.header("🎛️ Category Filter")
    try:
        df = load_customer_support_df(config.CUSTOMER_SUPPORT_CSV_PATH)
        categories = sorted(df["category"].unique().tolist())
    except Exception as e:
        st.error(f"Dataset error: {e}")
        st.stop()

    selected_categories = st.multiselect(
        "Choose categories",
        options=categories,
        default=categories,
        help="Filter the dataset used for retrieval.",
    )

    st.header("🧩 Sample Questions")
    sample_questions = [
        "How can I track my order?",
        "What is the return process?",
        "How do I request a refund?",
        "I forgot my password. What should I do?",
        "How do I update my shipping address?",
        "Can I cancel my order?",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()

    st.markdown("---")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.pending_question = None
        st.rerun()

# Main chat interface
st.markdown("---")
st.subheader("💬 Customer Support Q&A")
st.caption("Type your question in the chat box below (filtered by selected categories).")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User satisfaction score (subjective, collected via UI)
assistant_count = sum(1 for m in st.session_state.chat_history if m.get("role") == "assistant")
if assistant_count > 0:
    assistant_index = str(assistant_count - 1)
    rating_options = ["Not rated", "Helpful", "Not helpful"]

    existing = st.session_state.satisfaction.get(assistant_index)
    default_choice = "Not rated"
    if existing is not None:
        default_choice = "Helpful" if existing >= 0.5 else "Not helpful"

    rating_choice = st.selectbox(
        "User Satisfaction Score (proxy)",
        rating_options,
        index=rating_options.index(default_choice),
        key=f"satisfaction_{assistant_index}",
    )

    if rating_choice == "Helpful":
        st.session_state.satisfaction[assistant_index] = 1.0
    elif rating_choice == "Not helpful":
        st.session_state.satisfaction[assistant_index] = 0.0
    else:
        st.session_state.satisfaction.pop(assistant_index, None)

    satisfaction_values = list(st.session_state.satisfaction.values())
    satisfaction_avg = (
        sum(satisfaction_values) / max(len(satisfaction_values), 1)
        if satisfaction_values
        else None
    )
    with st.expander("🧑‍💼 User Satisfaction Summary"):
        st.write(
            {
                "Average satisfaction (proxy)": None if satisfaction_avg is None else round(satisfaction_avg, 3),
                "Ratings collected": len(satisfaction_values),
            }
        )

# If user clicked a sample question, process it as if it was submitted.
question_from_button = st.session_state.pending_question
if question_from_button is not None:
    st.session_state.pending_question = None
    question = question_from_button
else:
    question = st.chat_input("Ask about orders / refund / login ...")

if question:
    # Add user message to chat
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                selected = selected_categories if selected_categories else categories
                vectorstore, from_disk = get_vectorstore_for_categories(tuple(selected))

                result = answer_with_metrics(question, vectorstore)
                answer = result["answer"]

                st.markdown(answer)
                if config.SHOW_SOURCES:
                    with st.expander("📚 View Sources"):
                        for i, doc in enumerate(result["source_documents"], 1):
                            st.markdown(f"**Source {i} (Category: {doc.metadata.get('category', 'General')}):**")
                            preview = truncate_text(doc.page_content, config.SOURCE_PREVIEW_LENGTH)
                            st.text(preview)
                            st.markdown("---")

                if config.SHOW_EVALUATION_METRICS:
                    m = result["metrics"]
                    with st.expander("📊 Evaluation Metrics"):
                        st.write(
                            {
                                "Retrieval Latency (s)": round(m["retrieval_latency_seconds"], 4),
                                "Context Relevance (proxy)": round(m["context_relevance"], 4),
                                "Response Accuracy (proxy F1)": round(m["response_accuracy_proxy_f1"], 4),
                                "Precision": round(m["precision"], 4),
                                "Recall": round(m["recall"], 4),
                                "Generation Latency (s)": round(m["generation_latency_seconds"], 4),
                                "Vector Store": "Loaded from disk" if from_disk else "Built new",
                            }
                        )

                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                logger.info("Query processed successfully")

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                logger.error(f"Query processing error: {str(e)}")
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p style='color: #888; font-size: 0.9em;'>
            Built with LangChain, Streamlit, and Gemini
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
