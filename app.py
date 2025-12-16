import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from io import BytesIO

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from pypdf import PdfReader

import chromadb
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .env")

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHROMA_URL = os.getenv("CHROMA_URL", "http://127.0.0.1:8000")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "candidate_kb")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

def _parse_chroma(url: str):
    u = urlparse(url)
    return (u.hostname or "localhost"), (u.port or 8000)

host, port = _parse_chroma(CHROMA_URL)
chroma = chromadb.HttpClient(host=host, port=port)
collection = chroma.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

app = FastAPI()

def clean_text(t: str) -> str:
    t = t.replace("\x00", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def chunk_text(t: str, size: int = 2200, overlap: int = 250) -> List[str]:
    t = clean_text(t)
    if not t:
        return []
    out = []
    i = 0
    while i < len(t):
        j = min(i + size, len(t))
        piece = t[i:j].strip()
        if piece:
            out.append(piece)
        if j == len(t):
            break
        i = max(j - overlap, 0)
    return out

def embed(texts: List[str]) -> List[List[float]]:
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
    resp_sorted = sorted(resp.data, key=lambda x: x.index)
    return [x.embedding for x in resp_sorted]

@app.get("/health")
def health():
    return {"ok": True, "collection": COLLECTION_NAME, "chroma": CHROMA_URL}

@app.post("/ingest/pdf")
async def ingest_pdf(
    file: UploadFile = File(...),
    candidate_id: str = Form(...),
    doc_id: str = Form(...),
    source_name: str = Form(...),
    org_id: Optional[str] = Form(None),
    doc_type: str = Form("cv_pdf"),
):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(400, "Only PDF supported")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(400, "Empty PDF")

    reader = PdfReader(BytesIO(pdf_bytes))
    ids, docs, metas = [], [], []

    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = clean_text(text)
        if not text:
            continue

        chunks = chunk_text(text)
        for ci, ch in enumerate(chunks):
            chunk_id = f"cand{candidate_id}_doc{doc_id}_p{page_num}_c{ci}"
            ids.append(chunk_id)
            docs.append(ch)
            metas.append({
                "org_id": org_id,
                "candidate_id": candidate_id,
                "doc_id": doc_id,
                "doc_type": doc_type,
                "source_name": source_name,
                "page": page_num,
                "chunk_index": ci,
            })

    if not docs:
        return {"ok": True, "chunks_added": 0, "warning": "No extractable text"}

    vectors = embed(docs)

    collection.upsert(
        ids=ids,
        documents=docs,
        metadatas=metas,
        embeddings=vectors,
    )

    return {"ok": True, "chunks_added": len(docs)}

@app.post("/query")
async def query(payload: Dict[str, Any]):
    candidate_id = str(payload.get("candidate_id", "")).strip()
    question = str(payload.get("question", "")).strip()
    org_id = payload.get("org_id", None)
    top_k = int(payload.get("top_k", 8))

    if not candidate_id or not question:
        raise HTTPException(400, "candidate_id and question are required")

    where = {"candidate_id": candidate_id}
    if org_id is not None:
        where["org_id"] = str(org_id)

    qv = embed([question])[0]

    res = collection.query(
        query_embeddings=[qv],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    ids = (res.get("ids") or [[]])[0]

    hits = []
    for i in range(len(docs)):
        m = metas[i] or {}
        hits.append({
            "chunk_id": ids[i],
            "text": docs[i],
            "source_name": m.get("source_name"),
            "page": m.get("page"),
            "doc_id": m.get("doc_id"),
            "doc_type": m.get("doc_type"),
        })

    return {"ok": True, "hits": hits}
