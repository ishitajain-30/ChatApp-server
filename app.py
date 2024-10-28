from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import Optional

from utils import Chatbot, Embedder

app = FastAPI()

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity; restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot = Chatbot(model_name="llama3-8b-8192")  # Initialize the Chatbot with your Groq model
embedder = Embedder()  # Initialize the Embedder

current_vectors = None  # Variable to store the embeddings

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile):
    """
    Endpoint for uploading a PDF document.
    """
    global current_vectors

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File type must be PDF")

    original_filename = file.filename
    file_content = await file.read()

    try:
        # Process the uploaded file and store embeddings
        current_vectors = embedder.get_doc_embeds(file_content, original_filename)
        return JSONResponse(content={"message": f"Embeddings for {original_filename} have been loaded."})
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

@app.post("/ask/")
async def ask_question(query: str = Form(...)):
    """
    Endpoint for answering questions based on the uploaded PDF content.
    """
    global current_vectors

    if current_vectors is None:
        raise HTTPException(status_code=400, detail="No PDF embeddings available. Please upload a PDF first.")

    try:
        # Use the conversational chat function to get the response
        response = chatbot.conversational_chat(query, vectors=current_vectors)
        return JSONResponse(content={"answer": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred while processing the question.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
