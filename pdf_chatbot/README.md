# Local PDF Assistant (Mistral 7B + Ollama)

A powerful, fully offline chatbot that answers questions based on your PDF documents using the Mistral 7B model via [Ollama](https://ollama.ai). It's fast, private, and cost-free — designed for working with technical, academic, and business documents.

---

## Key Features

* **Runs Locally**: Uses Mistral 7B on your own machine via Ollama.
* **Fully Offline**: No internet connection required after setup.
* **No API Keys or Fees**: Zero cost, unlimited use.
* **Semantic Search**: Understands meaning beyond keywords.
* **Accurate Source Referencing**: Answers cite PDF names and page numbers.
* **Conversational Memory**: Maintains context throughout the session.
* **Scalable**: Efficiently processes and searches across thousands of PDFs.

---

## Quick Start Guide

### 1. Install Ollama and Mistral

Install [Ollama](https://ollama.ai) and pull the Mistral model:

```bash
ollama pull mistral
ollama serve  # Leave this running
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Your PDF Files

Place your documents into the `data/database/manual/fr/` folder:

```bash
cp your_documents/*.pdf data/database/manual/fr/
```

### 4. Process the PDFs

This step prepares your documents for search:

```bash
python database_processor.py
```

### 5. Start the Chat Interface

Begin asking questions about your PDFs:

```bash
python chat_agent.py
```

---

## Example Conversation

```
You: What welding equipment is listed in the documents?
Assistant: The manuals mention several models in the TITANIUM series, such as the TITANIUM 160, 200, 230 AC/DC FV, and 250. These support various welding processes including TIG and MIG.

You: Who manufactures them?
Assistant: These are produced by GYS, a French company headquartered in Saint-Berthevin, with international operations.
```

---

## Project Layout

```
pdf_chatbot/
├── chat_agent.py            # Chat interface using Mistral 7B
├── database_processor.py    # PDF indexing and embedding
├── requirements.txt         # Python dependencies
├── data/
│   ├── database/
│   │   └── manual/
│   │       └── fr/          # Your French PDF documents
│   └── Vectorized/          # Local vector index (FAISS)
└── README.md                # This file
```

---

## How It Works

1. **Text Extraction** – Parses and extracts text from PDFs.
2. **Chunking** – Splits documents into manageable text segments.
3. **Embedding** – Creates vector representations of the text using `sentence-transformers`.
4. **Vector Indexing** – Stores vectors using FAISS for fast search.
5. **Semantic Search** – Retrieves relevant content based on meaning.
6. **Answer Generation** – Mistral 7B analyzes the results and formulates answers.

---

## Configuration Options

### Change the Model

Use an alternative model supported by Ollama:

```bash
python chat_agent.py --model llama2
python chat_agent.py --model codellama
```

### Custom Paths

Specify custom locations for PDFs or output data:

```bash
python database_processor.py --database-dir /path/to/database --output-dir /path/to/vectorstore
python chat_agent.py --pdf-dir /path/to/database/manual/fr --vector-dir /path/to/vectorstore
```

### Force Vector Store Rebuild

Reprocess all documents from scratch:

```bash
python database_processor.py --force
```

---

## Search Capabilities

The assistant uses intelligent semantic search to:

* Recognize synonyms and related terms (e.g., “model” vs. “type”)
* Understand context and make logical inferences
* Prioritize relevant content over boilerplate
* Synthesize answers from multiple sources when necessary

---

## System Requirements

* **Python 3.8 or newer**
* **Ollama with Mistral 7B**
* **Minimum 8GB RAM recommended**
* **Local storage** for vector database and PDF files

For low-memory machines, try:

```bash
ollama pull mistral:7b-instruct-q4_0
```

---

## Performance

* **PDF processing**: \~2–5 files per second (initial run)
* **Search speed**: Instant after indexing
* **Memory usage**: Efficient, with overlapping chunking
* **Storage**: Vector index is cached locally for reuse

---

## Privacy and Security

* All data remains on your machine.
* No cloud dependencies or API calls.
* No telemetry or tracking of any kind.
* Designed for private, secure document handling.

---

## Troubleshooting

**Ollama not responding?**

```bash
ollama list
ollama serve  # Start Ollama if it's not running
```

**Model missing?**

```bash
ollama pull mistral
```

**No PDFs detected?**

```bash
ls data/database/manual/fr/  # Confirm PDFs are in the correct folder
```

**Search results missing or outdated?**

```bash
python database_processor.py --force
```

---

**Fully offline. No subscription. No compromises on privacy. Just fast, intelligent answers from your PDFs.**

---