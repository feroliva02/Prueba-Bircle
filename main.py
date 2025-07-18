#FastAPI app y endpoints
from fastapi import FastAPI, HTTPException
from rag import query_documents, generate_answer
from orchestrator import agent_query

app = FastAPI()

#Recibimos una pregunta (q) por query, chequeamos que q existe y buscamos en Chroma las 3 respuestas mas relevantes con la query_documents(), 
#devolvemos JSON

@app.get("/query")              #Aca le decimos que cuando llegue un get/query, ejecute la funcion de abajo query()
def query(q: str):
  if not q:                     #Revisa que haya query (q)
    raise HTTPException(status_code=400, detail="El parametro query 'q' es requerido")
  results = query_documents(q, k=3)
  documentos = results["documents"][0] #Es una lista dentro de una lista [[]]

  if not documentos:            #Revisa que haya documentos
    raise HTTPException(status_code=404, detail="No se encontraron documentos")

  respuesta = generate_answer(q, documentos)

  return{
    "query": q,
    "documents": documentos,
    "answer": respuesta
  }

@app.get("/agent_query")       #Aca le decimos que cuando llegue un get/agent_query, ejecute la funcion de abajo agent_query_endpoint()
def agent_query_endpoint(q: str):
  if not q:                    #Revisa que haya query (q)
    raise HTTPException(status_code=400, detail="El parametro query 'q' es requerido")
  respuesta = agent_query(q)

  if not respuesta:           #Revisa que haya respuesta
    raise HTTPException(status_code=404, detail="No se encontro respuesta")

  return {
    "query": q,
    "answer": respuesta
  }