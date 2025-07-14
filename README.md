# Local PDF Assistant (Mistral 7B + Ollama)

A powerful, fully offline chatbot that answers questions based on your PDF documents using the Mistral 7B model via [Ollama](https://ollama.ai). Features intelligent French content extraction from multilingual documents and dynamic language detection for truly multilingual conversations.

---

## Key Features

* **Runs Locally**: Uses Mistral 7B on your own machine via Ollama.
* **Fully Offline**: No internet connection required after setup.
* **No API Keys or Fees**: Zero cost, unlimited use.
* **Multilingual Support**: Automatically detects and responds in French, English, Spanish, German, or Italian.
* **French Content Extraction**: For now only extracts french content to avoid duplicates and then translates.
* **Semantic Search**: Understands meaning beyond keywords with hybrid search.
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

This step extracts French content and prepares documents for search:

```bash
python database_processor.py
```

### 5. Start the Chat Interface

Begin asking questions in any supported language:

```bash
python chat_agent.py
```

---

## Example Conversations

### English
```
You: What welding equipment is listed in the documents?
Assistant: The manuals mention several models in the TITANIUM series, such as the TITANIUM 160, 200, 230 AC/DC FV, and 250. These support various welding processes including TIG and MIG.
```

---

## Project Layout

```
GYS/
├── chat_agent.py            # Chat interface using Mistral 7B
├── database_processor.py    # PDF processing with French extraction
├── requirements.txt         # Python dependencies (includes langdetect)
├── data/
│   ├── database/
│   │   └── manual/
│   │       └── fr/          # Your PDF documents (any language)
│   └── Vectorized/          # Local vector index (FAISS)
└── README.md                # This file
```

---

## How It Works

1. **Language Detection** – Uses `langdetect` to identify French content in multilingual PDFs.
2. **French Extraction** – Extracts only French sections from each document page.
3. **Text Chunking** – Splits documents into manageable, searchable segments.
4. **Embedding** – Creates vector representations using optimized models.
5. **Vector Indexing** – Stores vectors using FAISS for lightning-fast search.
6. **Hybrid Search** – Combines semantic similarity with keyword matching.
7. **Dynamic Language Response** – Detects user's language and responds accordingly.
8. **Answer Generation** – Mistral 7B analyzes results and formulates contextual answers.

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

### GPU/CPU Configuration

Control hardware usage:

```bash
python database_processor.py --no-gpu  # Force CPU usage
python database_processor.py --batch-size 128  # Adjust batch size
```

---

## Language Support

### Input Languages Supported
- **French** (français)
- **English** 
- **Spanish** (español)
- **German** (Deutsch)
- **Italian** (italiano)

### PDF Processing
- Automatically extracts **French content only** from multilingual PDFs
- Uses advanced language detection to identify language boundaries
- Preserves technical content even when language detection is uncertain

---

## Search Capabilities

The assistant uses intelligent hybrid search to:

* **Semantic Understanding**: Recognize synonyms and related terms
* **Keyword Matching**: Find exact technical terms and model numbers
* **Context Awareness**: Understand follow-up questions naturally
* **Source Prioritization**: Prefer recently referenced documents
* **Content Ranking**: Combine multiple scoring methods for best results

---

## System Requirements

* **Python 3.8 or newer**
* **Ollama with Mistral 7B**
* **Minimum 8GB RAM recommended** (16GB for GPU acceleration)
* **Local storage** for vector database and PDF files

### GPU Acceleration (Optional)
For faster processing with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For low-memory machines, try the quantized model:
```bash
ollama pull mistral:7b-instruct-q4_0
```

---

## Performance

* **PDF processing**: ~2–5 files per second
* **Search speed**: Instant after indexing
* **Memory usage**: Optimized chunking with 200-character overlap
* **Storage**: Vector index cached locally for reuse
* **Language detection**: Real-time for chat responses

---

## Privacy and Security

* **100% Local**: All data remains on your machine
* **No Cloud Dependencies**: No internet required after setup
* **No Telemetry**: No tracking or data collection
* **Secure Processing**: Documents never leave your system
* **Private Conversations**: Chat history stored locally only

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

**Language detection issues?**
```bash
pip install --upgrade langdetect
```

**GPU memory errors?**
```bash
python database_processor.py --no-gpu --batch-size 64
```

---

**Fully offline. Multilingual. No subscription. No compromises on privacy. Just fast, intelligent answers from your PDFs in any supported language.**

---