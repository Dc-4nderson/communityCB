# CIC Community Chatbot

A Streamlit-powered chatbot for the CIC community that combines speech-to-text, Retrieval-Augmented Generation (RAG) with Pinecone, and response generation using Meta Llama 3.2 3B Instruct (via the Hugging Face Inference API).

## Features

- **Document ingestion (admin only):** Upload a PDF, extract its text, chunk it semantically, embed it, and upsert it into a Pinecone index — gated behind an admin password
- **Text Q&A:** Ask a question in a form and get an answer grounded in the retrieved context
- **Voice Q&A:** Upload an audio file or record a question directly in the browser; audio is transcribed with IBM Watson Speech to Text before being answered
- **RAG-powered retrieval:** Relevant chunks are pulled from Pinecone using `sentence-transformers` (`all-MiniLM-L6-v2`) embeddings
- **Grounded response generation:** Llama 3.2 3B Instruct is prompted to answer only from retrieved context, and to say so explicitly when context is missing or doesn't fit
- **Secure config:** All API keys and secrets are loaded from environment variables

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Dc-4nderson/communityCB.git
cd communityCB
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

You'll also need `ffmpeg` installed and available for audio processing (via `pydub`).

### 3. Set up environment variables

Create an env file (referenced in code as `Key.env`) with:

```
HUG_TOKEN=your-huggingface-token
IBM_API_KEY=your-ibm-key
IBM_URL=your-ibm-url
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=your-pinecone-index
RAG_FILE=your-rag-file-path
ADMIN_PSW=your-admin-password
```

> **Note:** the `.env` path and the local `ffmpeg` path are currently hardcoded to Windows paths in `test2.py` (`load_dotenv(dotenv_path="C:\\codes\\testStreamlit\\Key.env")` and `pydub.AudioSegment.ffmpeg = "C:/ffmpeg/bin/ffmpeg.exe"`). Update these to match your local setup/OS before running.

### 4. Run the app

```bash
streamlit run test2.py
```

The app is configured to run on port `8501` (see `config.toml`).

## How it works

1. **(Admin) Upsert data:** Enter the admin password, upload a PDF, and click "Upsert Data." The document is chunked, embedded, and stored in Pinecone with an AI-generated chunk name.
2. **Ask a question:** Type a question in the text form, or upload/record audio (transcribed via IBM Watson).
3. **Retrieve + generate:** The top-matching chunks are pulled from Pinecone and passed as context to Llama 3.2 3B Instruct, which is prompted to answer strictly from that context — or to flag when it's guessing.

## Tech stack

| Component            | Tool                                   |
|-----------------------|-----------------------------------------|
| UI                    | Streamlit                              |
| Document parsing      | `pdfplumber`, `python-docx`            |
| Speech-to-text        | IBM Watson Speech to Text              |
| Embeddings            | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector store           | Pinecone                               |
| Response generation   | Meta Llama 3.2 3B Instruct (Hugging Face Inference API) |

## Configuration reference

`config.toml` sets the Streamlit server (headless, port `8501`, CORS disabled) and a dark theme with a green (`#4CAF50`) accent.
