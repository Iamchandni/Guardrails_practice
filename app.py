from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain_classic.prompts import PromptTemplate

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,AutoModelForCausalLM, pipeline
import re

# ==========================================================
# STEP 1: Load & Sanitize Corpus
# ==========================================================
with open("documents/data.txt", "r", encoding="utf-8") as f:
    corpus = f.read()

def sanitize_context(text):
    patterns = {
        r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b': '[REDACTED_EMAIL]',
        r'\b\d{10}\b': '[REDACTED_PHONE]',
        r'(?:\d[ -]*?){13,16}': '[REDACTED_CARD]',
        r'AKIA[0-9A-Z]{16}': '[REDACTED_AWS_KEY]',
        r'(?i)password\s*[:=]\s*\S+': '[REDACTED_PASSWORD]',
        r'(?i)api[_-]?key\s*[:=]\s*\S+': '[REDACTED_API_KEY]',
        r'\bIFSC:\s*\w+\b': '[REDACTED_IFSC]',
        r'\bCL-\d{5}\b': '[REDACTED_CLIENT_ID]'
    }
    for pattern, replacement in patterns.items():
        text = re.sub(pattern, replacement, text)
    return text

corpus_clean = sanitize_context(corpus)

# ==========================================================
# STEP 2: Split & Embed (optimized)
# ==========================================================
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.create_documents([corpus_clean])

# Smaller embedding model for CPU efficiency
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

vectorstore = FAISS.from_documents(docs, embeddings)

# ==========================================================
# STEP 3: Load local lightweight LLM (TinyLlama)
# ==========================================================
print("⏳ Loading TinyLlama model (optimized for CPU)...")

MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    temperature=0.3,
)

# ==========================================================
# STEP 4: Guardrail Prompt
# ==========================================================
def build_prompt(context, question):
    return f"""
You are an AI assistant with access to company documentation.
Follow these rules:
1. Never reveal any sensitive or redacted data.
2. If the question requests such data, respond with:
   "I'm sorry, but I can’t share that information as it may contain confidential data."
3. Provide short, clear, and relevant responses.

### Context:
{context}

### Question:
{question}

### Response:
"""

# ==========================================================
# STEP 5: Query Loop
# ==========================================================
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

while True:
    query = input("\n❓ Ask a question (or type 'exit'): ").strip()
    if query.lower() == "exit":
        break

    context_docs = retriever.invoke(query)
    context_text = "\n".join([doc.page_content for doc in context_docs])

    prompt = build_prompt(context_text, query)

    output = pipe(prompt, do_sample=True, temperature=0.3)[0]["generated_text"]

    if "### Response:" in output:
        response = output.split("### Response:")[-1].strip()
    else:
        response = output.strip()
        if not response:
          response = "I'm sorry, but I couldn’t find any non-sensitive information related to that query."

    

    print("\n💬 Answer:", response)
