# 🤖 RAG-Based Customer Service Chatbot using Gemini

A **production-grade** Retrieval-Augmented Generation (RAG) chatbot that answers customer support questions using a structured `customer_support.csv` dataset (orders / refunds / login, etc.) with Gemini + real HuggingFace embeddings.

This project focuses on building a smart customer support chatbot using a Retrieval-Augmented Generation (RAG) pipeline powered by the Gemini LLM. The goal is to provide fast, accurate, and context-aware responses by retrieving relevant information from a company knowledge base instead of relying on static FAQ systems.

## 🔹 Key Objectives

- Implement a RAG architecture for intelligent query handling
- Integrate Gemini LLM for human-like response generation
- Use vector databases (FAISS / Pinecone / ChromaDB) for similarity search
- Build a chatbot interface using Streamlit or Flask
- Evaluate chatbot performance using metrics like accuracy, latency, and user satisfaction

## 🔹 Project Approach

- **Data Collection**: Gather FAQs, chat logs, manuals, or support documents.
- **Preprocessing**: Clean and chunk text data for better retrieval.
- **Embedding Generation**: Convert text into vector embeddings using sentence transformer models.
- **Vector Storage**: Store embeddings in a vector database.
- **Query Processing**: Retrieve relevant context based on user queries.
- **Response Generation**: Use Gemini LLM to generate contextual answers.
- **Frontend Development**: Create a chatbot UI for interaction.
- **Evaluation**: Measure response relevance, speed, and overall performance.

## 🔹 Business Use Cases

- Automating customer support queries
- Assisting e-commerce customers (order tracking, returns)
- Banking helpdesk (balance, loans, card issues)
- Travel support (ticket details, refunds)
- IT helpdesk query resolution

## 🔹 Expected Outcomes

- Build a fully functional domain-specific RAG chatbot
- Improve response accuracy and reduce latency compared to traditional FAQ bots
- Enable real-world deployment in customer service environments
- Gain hands-on skills in LLMs, NLP, vector search, prompt engineering, and chatbot deployment

## ✨ Features

- **Customer Support Dataset**: Uses `customer_support.csv` (question/answer/category) as the single data source
- **Gemini LLM**: Uses `langchain-google-genai` for response generation
- **Category Filtering**: Filter retrieval by `Order`, `Refund`, `Return`, `Login`, etc.
- **Sample Questions**: One-click prompts for common support intents
- **Chat History**: Conversation persists during the Streamlit session
- **Evaluation Metrics (UI)**: Retrieval latency, context relevance, response accuracy proxy (F1), and Precision/Recall
- **User Satisfaction Score (UI)**: Optional subjective helpfulness rating collected after each answer
- **Persistent Vector DB**: FAISS indexes are saved to disk and reloaded on demand
- **Source Citations**: Shows retrieved context passages used for the answer

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google (Gemini) API key as `GOOGLE_API_KEY` (from [ai.google.dev](https://ai.google.dev/))

### Installation

```bash
# 1. Clone/download the project
cd "RAG-Based Customer Service Chatbot using Gemini (Gen AI)"

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# 5. Run the application
streamlit run rag_chatbot.py
```

The app opens at `http://localhost:8501`

## 📖 Usage
1. **Filter Categories**: Choose `Order` / `Refund` / `Return` / `Login` (etc.) in the sidebar
2. **Ask a Question**: Type your query or click a sample question
3. **Cite & Evaluate**: Expand `View Sources` and `Evaluation Metrics` to see retrieval + proxy accuracy

## 🏗️ Project Structure

```
RAG-Based Customer Service Chatbot using Gemini (Gen AI)/
├── rag_chatbot.py          # Main Streamlit application
├── config.py               # Configuration management
├── utils.py                # Utility functions
├── requirements.txt        # Dependencies
├── .env.example            # Environment template
├── .gitignore              # Git ignore rules  
└── README.md               # This file
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Document Processing
CHUNK_SIZE = 1000           # Size of text chunks
CHUNK_OVERLAP = 200         # Overlap between chunks

# Retrieval 
RETRIEVAL_K = 3             # Number of sources to retrieve

# Language Model
LLM_TEMPERATURE = 0.0       # 0 = deterministic, 1 = random
GEMINI_MODEL = "gemini-flash-latest"

# Logging
LOG_LEVEL = "INFO"          # DEBUG, INFO, WARNING, ERROR
```

## 📊 How It Works

```
customer_support.csv → Read (question/answer/category) → Embed → Store in FAISS
                                                 ↓
                            User → Query → Retrieve Similar → Gemini → Answer
```

1. **Load**: Read `customer_support.csv` (question/answer/category)
2. **Chunk**: Answers are split into smaller passages for retrieval
3. **Embed**: Chunks converted to vector embeddings using HuggingFace
4. **Store**: Vectors indexed in FAISS (saved locally for persistence)
5. **Filter/Retrieve**: Select categories, then retrieve the most similar chunks
6. **Generate**: Gemini generates a response strictly from retrieved context

## 🧠 System Architecture Diagram

```mermaid
flowchart LR
    U[User] --> Q[Query]
    Q --> E[Embed (HuggingFace)]
    E --> R[Retrieve (FAISS)]
    R --> G[Gemini LLM]
    G --> A[Response]
```

## 🏬 Real Business Use Case

This chatbot helps e-commerce customers resolve common support requests automatically by answering from an internal knowledge dataset:
- Track orders and shipment status
- Guide refund/return workflows and timelines
- Fix login issues (forgot password, email updates)

It also exposes retrieval latency, context relevance, and a response accuracy proxy to support evaluation during demos and final submissions.

## 🔑 Getting API Key
1. Visit [ai.google.dev](https://ai.google.dev/)
2. Create/enable a Gemini API key
3. Add to `.env`: `GOOGLE_API_KEY=your_key_here`

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| API Key not found | Check `.env` exists with valid key |
| File loading error | Verify PDF/TXT/DOCX format is valid |
| Slow processing | Try smaller files or increase CHUNK_SIZE |
| Import errors | Run `pip install -r requirements.txt` |
| Memory issues | Process fewer files at once |

## 📁 Dataset Support

| Format | File | Status |
|--------|------|--------|
| CSV | `customer_support.csv` | ✅ Required |

## 🚀 Deployment Notes

## 🧾 Report (Summary of Results)

For your submission report, capture results from the app’s UI:

- For 5-10 test questions (use the sample questions + your own), record:
  - `Retrieval Latency (s)` (FAISS similarity search time)
  - `Generation Latency (s)` (Gemini response generation time)
  - `Context Relevance (proxy)` (FAISS score-based proxy)
  - `Response Accuracy (proxy F1)` (token overlap vs. expected answer)
  - `Precision` and `Recall` (retrieval proxy)
  - `User Satisfaction Score (proxy)` (Helpful = 1, Not helpful = 0, average shown in the UI)

- Add 1-2 lines on observed strengths/weaknesses (e.g., which categories answered best).

- Include screenshots of:
  - The chatbot conversation
  - The `📊 Evaluation Metrics` expander
  - The `🧑‍💼 User Satisfaction Summary` expander

### Logging
Application includes comprehensive logging:
- Document processing steps
- Query handling
- Error details and stack traces

### Error Handling
- Graceful error recovery
- User-friendly error messages
- Automatic cleanup of temporary files
- File encoding fallback (UTF-8 → Latin-1)

### Performance Characteristics
- Persistent FAISS vector indexes saved to disk and reloaded on demand
- Real semantic embeddings using `HuggingFaceEmbeddings` (sentence-transformers)
- Streamlit caching for faster subsequent queries (dataset + embeddings + indexes)

## 🔐 Security

- API keys only in `.env` (not in code)
- Dataset schema validated at startup (question/answer/category)
- No sensitive data in logs
- `.gitignore` prevents key exposure

## 📚 Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| UI | Streamlit | Web interface |
| RAG | LangChain | Orchestration |
| LLM | Gemini API (via `langchain-google-genai`) | Language model |
| Vectors | FAISS | Similarity search |
| Embeddings | HuggingFaceEmbeddings | Vector representation |
| Data | `customer_support.csv` | Customer support answers (question/answer/category) |

## 🔄 Embeddings Configuration

This project uses real HuggingFace sentence-transformers embeddings:
`sentence-transformers/all-MiniLM-L6-v2`.

FAISS indexes are persisted to `vector_store/` and reloaded to improve production readiness.

## 📖 Dependencies

Core packages:
- `streamlit>=1.28.0` - UI framework
- `langchain>=0.1.0` - RAG framework
- `langchain-google-genai` - Gemini integration
- `faiss-cpu>=1.7.4` - Vector search
- `python-dotenv>=1.0.0` - Environment management
- `pandas>=2.0.0` - Dataset handling
- `sentence-transformers` - Real embeddings

See `requirements.txt` for complete list.

## 🧪 Development

### Running in Debug Mode
```bash
# Set environment variable
set ENVIRONMENT=development  # Windows
export ENVIRONMENT=development  # Linux/Mac

# Run app
streamlit run rag_chatbot.py
```

### Code Quality
- Full type hints for all functions
- Comprehensive docstrings
- Proper error handling
- Logging at appropriate levels

## 📝 Example Queries

Try a few intent-based queries:
- "What are the main topics covered?"
- "Summarize the key findings"
- "What does it say about [topic]?"
- "Compare X and Y"
- "What are the conclusions?"

## 🎯 Best Practices

- **Chunk Size**: 1000-2000 tokens for balanced context
- **Temperature**: 0-0.3 for factual answers
- **Retrieval Count**: 3-5 for balance between relevance and context
- **File Size**: Keep documents under 100MB
- **Encoding**: UTF-8 preferred, auto-fallback to Latin-1

## 🆘 Getting Help

1. Check Troubleshooting section
2. Review logs: `ENVIRONMENT=development streamlit run rag_chatbot.py`
3. Verify `.env` configuration
4. Check file format compatibility
5. Review [LangChain documentation](https://python.langchain.com)

## 📄 License

MIT License - Use freely and modify as needed

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional file format support (PowerPoint, Excel)
- Real embedding models integration
- Vector database persistence
- Advanced retrieval strategies
- Performance benchmarking

---

**Built with** ❤️ using LangChain, Streamlit, and Gemini

Last Updated: March 2026
