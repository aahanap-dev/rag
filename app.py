import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from fastapi.responses import Response

from RAGService import RAGInitializer
import logging
import traceback
from fastapi.responses import JSONResponse


# Configure Logger for API
logging.basicConfig(
    filename="rag_api.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize FastAPI app
app = FastAPI(
    title="RAG API",
    description="API for querying Ubuntu docs using RAG.",
    version="1.0.0"
)

try:
    # Initialize RAG system
    logging.info("Initializing RAG system for API...")
    rag_system = RAGInitializer()
    logging.info("RAG system initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing RAG system: {e}")
    logging.debug(traceback.format_exc())
    raise RuntimeError("Failed to initialize RAG system. Check logs for details.")


class QueryRequest(BaseModel):
    question: str


@app.post("/query_with_source", response_model=Dict[str, Any])
def process_query_source(request: QueryRequest):
    """
    Endpoint to process user queries via Ubuntu RAG.

    Input:
    - question: User query as a string.

    Output:
    - JSON response with the question, answer, and source documents.
    """
    try:
        logging.info(f"Received query: {request.question}")

        # Process query using RAG
        response = rag_system.process_query(request.question)

        # Prepare source document information
        sources = [
            f"File Name: {doc.metadata.get('source', 'Unknown').split('/')[-1]} Content: {doc.page_content}"
            for doc in response["source_documents"]
        ]

        # Prepare and return the response
        result = {
            "question": response["question"],
            "answer": response["answer"],
            "sources": sources,
        }

        logging.info("Query processed successfully.")
        return result

    except HTTPException as he:
        logging.error(f"HTTPException: {he.detail}")
        logging.debug(traceback.format_exc())
        raise he

    except Exception as e:
        logging.error(f"Error processing query: {e}")
        logging.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/query", response_model=Dict[str, Any])
def process_query(request: QueryRequest):
    """
    Endpoint to process user queries via Ubuntu RAG.

    Input:
    - question: User query as a string.

    Output:
    - JSON response with the question, answer
    """
    try:
        logging.info(f"Received query: {request.question}")

        # Process query using RAG
        response = rag_system.process_query(request.question)

        logging.info("Query processed successfully.")
        # Extract only the answer
        answer = response.get("answer", "No answer available.")

        # Pretty-print the answer
        pretty_response = json.dumps({"answer": answer}, indent=4, ensure_ascii=False)

        # Return only the answer as a pretty-printed response
        return Response(content=pretty_response, media_type="application/json")

    except HTTPException as he:
        logging.error(f"HTTPException: {he.detail}")
        logging.debug(traceback.format_exc())
        raise he

    except Exception as e:
        logging.error(f"Error processing query: {e}")
        logging.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/")
def read_root():
    """
    Root endpoint to check API status.
    """
    try:
        logging.info("Root endpoint accessed.")
        return {"message": "RAG API is running!"}
    except Exception as e:
        logging.error(f"Error accessing root endpoint: {e}")
        logging.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Error accessing root endpoint.")
