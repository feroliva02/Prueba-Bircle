#Logica de RAG con Chroma
import os
import chromadb
import openai
from chromadb.utils import embedding_functions

#Generamos embeddings usando la API de OpenAI
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
  api_key= os.getenv("OPENAI_API_KEY"), 
  model_name = 'text-embedding-ada-002')

#Configuramos API KEY y generamos una respuesta llamando a la api de OpenAI, tomando la primer opcion que da (choices[0]).
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_answer(question, documents):
  context_text = "\n\n".join([f"Documento {i+1}: {doc}" for i, doc in enumerate(documents)])
  prompt = f"""Usa la siguiente informacion para responder la pregunta de la mejor forma posible:

Informacion relevante:
{context_text}
Pregunta: {question}
Respuesta:"""
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=300,
    temperature=0.2,
  )

  answer = response.choices[0].message.content
  return answer

#Creamos cliente de Chroma y una coleccion para guardar los documentos 
client = chromadb.Client()
collection = client.get_or_create_collection(
  name='documentos', 
  embedding_function=openai_ef)

files = [f for f in os.listdir("data") if f.endswith(".txt")]

#Recorremos la lista de los archivos, open los abre, read los lee, se guardan en contenido y se agregan a la coleccion
for filename in files:
  path = os.path.join("data", filename)
  with open(path, 'r', encoding= 'utf-8') as f:
    contenido = f.read()
  collection.add(
    documents = [contenido],
    ids = [filename],
    metadatas = [{'nombre_archivo': filename}]
  )

#Funcion que recibe una pregunta y una cantidad K, y busca en la coleccion los resultados (K) mas relevantes
def query_documents(pregunta, k=3):
  resultados = collection.query(
    query_texts = [pregunta],
    n_results = k
  )
  return resultados
