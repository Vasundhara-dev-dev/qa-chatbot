import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai

df = pd.read_csv("Training Dataset.csv")
df.fillna("Unknown", inplace = True)

documents = df.apply(lambda row: ' | '.join([f"{col}: {row[col]}" for col in df.columns]), axis = 1).tolist()

model = SentenceTransformer('all-MiniLM-L6-v2')
document_embeddings = model.encode(documents)

dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(document_embeddings))

def retrieve_docs(query, k = 3):
  query_vec = model.encode([query])
  distances, indices = index.search(np.array(query_vec), k)
  return [documents[idx] for idx in indices[0]]

openai.api_key = "API Key"

def generate_answer(query):
  context_docs = retrieve_docs(query)
  context = "\n".join(context_docs)
  prompt = f"""You are a helpful assistant. Use the context below to answer the question. Context: \n{context}\nQuestion: {query}Answer: """
  response = openai.ChatCompletion.create(model = "gpt-3.5-turbo", messages = [{"role": "user", "content": prompt}])
  return response.choices[0].message["content"]

query = "What are the factors affecting loan approval?"
answer = generate_answer(query)
print("Answer: ", answer)
