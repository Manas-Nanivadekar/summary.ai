from rag_app import RAGApplication
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
rag = RAGApplication()

# Initialize the RAG system with the local PDF
PDF_PATH = "document.pdf"


# Model for the query request
class Query(BaseModel):
    question: str


@app.on_event("startup")
async def startup_event():
    try:
        if not rag.load_existing_vector_store():
            print(f"Processing PDF: {PDF_PATH}")
            num_chunks = rag.load_pdf(PDF_PATH)
            print(f"Processed {num_chunks} text chunks")
        else:
            print("Loaded existing vector store")
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        raise


@app.post("/query")
async def query_endpoint(query: Query):
    try:
        answer = rag.query(query.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# You can run this with: uvicorn main:app --reload
# uvicorn main:app --host 0.0.0.0 --port 8000
