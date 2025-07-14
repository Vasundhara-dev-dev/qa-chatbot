import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import openai
import os

df = pd.read_csv("Training Dataset.csv")
df.fillna("Unknown", inplace=True)

df.head()

documents = df.apply(lambda row: ' | '.join([f"{col}: {row[col]}" for col in df.columns]), axis=1).tolist()

embedder = SentenceTransformer("all-MiniLM-L6-v2")
document_embeddings = embedder.encode(documents, show_progress_bar=False)
document_embeddings = np.array(document_embeddings).astype("float32")

dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)

def retrieve_relevant_docs(query, top_k=3):
    query_vec = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, top_k)
    return [documents[idx] for idx in indices[0]]

from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()

def generate_answer(query, top_k=3):
    context_docs = retrieve_relevant_docs(query, top_k)
    context = "\n".join(context_docs)

    prompt = f"""
You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=300
    )

    return response.choices[0].message.content

user_query = "What income level is most likely to get loan approval?"
response = generate_answer(user_query)
print("Answer:\n", response)
