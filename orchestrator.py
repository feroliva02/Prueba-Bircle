import os
import chromadb
from llama_index import (GPTVectorStoreIndex, ServiceContext, LLMPredictor)
from llama_index.vector_stores import ChromaVectorStore
from langchain.chat_models import ChatOpenAI

#Variable de entorno para que se pueda usar
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

#creamos cliente y usamos la coleccion de rag
client = chromadb.Client()
collection = client.get_collection(name="documentos")

#Creamos vector y usamos la coleccion para poder buscar en los documentos
vector_store = ChromaVectorStore(collection=collection)

#llmPredictor va a usar 'gpt-3.5-turbo' para generar las respuestas
llm_predictor = LLMPredictor(
  llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
)

#Configuracion para LlamaIndex diciendole que use el predictor que cree antes
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

#Indice de Chroma que va a usar LlamaIndex 
index = GPTVectorStoreIndex.from_vector_store(
  vector_store,
  service_context=service_context
)

#Funcion que recibe una pregunta y responde usando el indice y devuelve una respuesta 
def agent_query(question):
  response = index.query(question)
  return response.response