
# RAG Application with API


## Features
- Upload PDF documents and index them for semantic search.
- Query your documents in natural language and get context-aware answers.
- Powered by FAISS + Sentence Transformers for efficient similarity search.
- Runs locally with Ollama, no external API keys required.


## Installation

```bash
git clone https://github.com/codersbranch/rag_app_with_api.git
cd rag_app_with_api
Set up virtual environment 
pip install -r requirements.txt
```

## Usage
create folder uploaded_pdfs , vector_data
upload your pdfs in the uploaded_pdfs with different categories
To build vector indexes
```bash
python save_data.py
```
Run the app:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Then open the provided URL in your browser.




