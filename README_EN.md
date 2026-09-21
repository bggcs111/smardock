# smardock · Document Q&A Tool

A personal document knowledge base: archive PDF / Word files and ask questions in natural language, with every answer citing its source. Privacy documents are fully masked — real data never leaves your machine.

![Home screenshot](docs/screenshot.png)

---

## ✨ Highlights

### 📚 Drop in documents, start asking
- Supports PDF, Word, and other common formats — parsed and indexed automatically
- Create multiple knowledge bases, organized by topic
- Ask questions about a specific document, too

### 🎯 Accurate answers you can trust
- Understands paraphrased questions and pinpoints exact terms like model numbers and IDs
- Every answer cites its source; click to view the snippet or download the original

### 🔐 Privacy documents stay private
- Choose "Privacy mode" when creating a knowledge base; documents are masked before processing
- Names, ID numbers, phone numbers, addresses — all replaced by placeholders; real data never leaves your machine
- Even filenames and paths are masked

### 💬 Conversations auto-saved, organized by topic
- Create, switch, and delete conversations with a clear history
- Records are saved automatically and restored on refresh
- Repeated questions reuse cached answers, saving time and cost

---

## 🚀 Getting Started

### Get the code

```bash
git clone https://github.com/bggcs111/smardock.git
cd smardock
```

### Prerequisites
- Python 3.10+
- An embedding API key (Aliyun Bailian DashScope `text-embedding-v3` by default)
- A chat model API key (e.g., `deepseek-chat`)

### Models & environment

This project cleanly separates local models from cloud APIs: **sensitive-information detection runs locally; retrieval and generation run in the cloud**.

**Local NER model** (listed in `requirements.txt`, with version constraints)

- Model: `damo/nlp_raner_named-entity-recognition_chinese-base-generic`, runs entirely locally and improves masking accuracy. CPU inference is enough; no GPU required.
- Not installed by default. Install via `requirements.txt` if needed; otherwise it gracefully falls back to regex + field-anchor masking.

### One-click launch

**Windows**
```bat
start.bat
```

**macOS / Linux**
```bash
bash start.sh
```

The script creates a virtual environment, installs dependencies, and generates `.env` from `.env.example`.

### Manual launch

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # fill in your API keys
python app.py
```

Open **http://127.0.0.1:7860**

### Three steps

1. **Create a knowledge base**: choose a mode (normal / privacy)
2. **Upload**: select the knowledge base and drop in PDF or DOCX files
3. **Ask**: ask questions in the chat box; superscripts like `¹` mark the source — click to view snippets or download originals

### Configuration

All settings live in `config.py` and can be overridden via `.env`:

| Variable | Description | Default |
|---|---|---|
| `DASHSCOPE_API_KEY` | Embedding API key | — |
| `DEEPSEEK_API_KEY` | Chat model API key | — |
| `EMBEDDING_PROVIDER` | `dashscope` / `openai` (any OpenAI-compatible endpoint) | `dashscope` |
| `PRIVACY_STRICT_REDACTION` | Keep placeholders in privacy-mode answers | `true` |
| `PRIVACY_MASK_SOURCE` | Mask source filenames in privacy mode | `true` |
| `SECTION_EXPAND` | Expand matched section | `true` |
| `RERANK_ENABLED` | Enable reranking | `true` |
| `PII_EXTRA_FIELDS` | Extra fields to mask (comma-separated, e.g. `性别,职业类别`) | empty |
| `SERVER_PORT` | Server port | `7860` |

---

## 🗺️ Roadmap

- [ ] **Quick search for uploaded documents**: search past uploads and jump to archived content
- [ ] **Quick snippet preview**: preview referenced document snippets with one click
- [ ] **Manual masking review**: manually add and confirm masked items to avoid false positives
- [ ] **Better image / table / formula understanding**: improve recognition accuracy and use it automatically in answers
- [ ] **Polished UI**: refine layout, colors, and interactions

---

## 🙏 Acknowledgements

Built on these excellent open-source projects:

| Project | Purpose |
|---|---|
| [Gradio](https://github.com/gradio-app/gradio) | Web UI |
| [ChromaDB](https://github.com/chroma-core/chroma) | Vector storage & retrieval |
| [PyMuPDF](https://github.com/pymupdf/PyMuPDF) | PDF parsing |
| [pymupdf4llm](https://github.com/pymupdf/RAG) | PDF → Markdown |
| [python-docx](https://github.com/python-openxml/python-docx) | Word parsing |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | Env loading |
| [OpenAI Python SDK](https://github.com/openai/openai-python) | Chat model calls |
| [ModelScope](https://github.com/modelscope/modelscope) | Local NER (RaNER) |
| [SQLite FTS5](https://www.sqlite.org/fts5.html) | Keyword index |

Also thanks to **DeepSeek** and **Aliyun Bailian** for model capabilities.

---

## 📬 Contact

- Author: lowkyner
- Email: bggcs111@163.com
- Issues: [submit an issue](https://github.com/bggcs111/smardock/issues)

> If you run into problems, please include your environment info (OS / Python version / reproduction steps).

---

## 📄 License

Licensed under the [MIT License](LICENSE) — free for commercial and non-commercial use.
