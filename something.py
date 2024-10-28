import os
import pickle
import tempfile
from groq import Groq
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class Chatbot:
    def __init__(self, model_name, temperature=0.7, vectors=None):
        self.model_name = model_name
        self.temperature = temperature
        self.vectors = vectors
        self.client = Groq(api_key="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")

    def conversational_chat(self, query, vectors=None):
        """
        Start a conversational chat with the Groq model, using relevant context from the embeddings.
        """
        context = ""
        if vectors:
            # Search for the most relevant chunks using FAISS
            relevant_docs = vectors.similarity_search(query, k=5)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Include the context in the chat request to the model
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Answer the query only is given context in detail."
                },
                {
                    "role": "user",
                    "content": f"Based on the given context answer the query Context: {context} Query: {query}"
                }
            ],
            # stream=True
            model=self.model_name
        )

        # Extract the answer from the response
        result = chat_completion.choices[0].message.content
        return result

class Embedder:
    def __init__(self):
        self.PATH = "embeddings"
        self.create_embeddings_dir()

    def create_embeddings_dir(self):
        """
        Creates a directory to store the embeddings vectors
        """
        if not os.path.exists(self.PATH):
            os.mkdir(self.PATH)

    def store_doc_embeds(self, file, original_filename):
        """
        Stores document embeddings using Langchain and FAISS
        """
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_file:
            tmp_file.write(file)
            tmp_file_path = tmp_file.name

        file_extension = os.path.splitext(original_filename)[1].lower()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=100,
            length_function=len,
        )

        # if file_extension == ".csv":
        #     loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
        #     data = loader.load()
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path=tmp_file_path)
            data = loader.load_and_split(text_splitter)
        # elif file_extension == ".txt":
        #     loader = TextLoader(file_path=tmp_file_path, encoding="utf-8")
        #     data = loader.load_and_split(text_splitter)
        else:
            raise ValueError("Unsupported file type")

        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vectors = FAISS.from_documents(data, embeddings)
        os.remove(tmp_file_path)

        # Save the vectors to a pickle file
        with open(f"{self.PATH}/{original_filename}.pkl", "wb") as f:
            pickle.dump(vectors, f)

    def get_doc_embeds(self, file, original_filename):
        """
        Retrieves document embeddings
        """
        if not os.path.isfile(f"{self.PATH}/{original_filename}.pkl"):
            self.store_doc_embeds(file, original_filename)

        # Load the vectors from the pickle file
        with open(f"{self.PATH}/{original_filename}.pkl", "rb") as f:
            vectors = pickle.load(f)

        return vectors

def main():
    chatbot = Chatbot(model_name="llama3-8b-8192")  # Replace with your Groq model name
    embedder = Embedder()

    print("Welcome to the Groq Chatbot CLI")
    current_vectors = None

    while True:
        command = input("Enter 'chat' to start chatting, 'embed' to process a document, or 'exit' to quit: ").strip().lower()

        if command == "exit":
            print("Goodbye!")
            break
        elif command == "chat":
            user_input = input("You: ")
            response = chatbot.conversational_chat(user_input, vectors=current_vectors)
            print(f"Robby: {response}")
        elif command == "embed":
            file_path = input("Enter the path to the file (pdf, txt, csv): ").strip()
            if not os.path.isfile(file_path):
                print(f"File not found: {file_path}")
                continue

            original_filename = os.path.basename(file_path)
            with open(file_path, "rb") as file:
                file_content = file.read()

            try:
                current_vectors = embedder.get_doc_embeds(file_content, original_filename)
                print(f"Embeddings for {original_filename} have been loaded and are now available for chat.")
            except ValueError as ve:
                print(f"Error: {str(ve)}")
        else:
            print("Invalid command. Please enter 'chat', 'embed', or 'exit'.")

if __name__ == "__main__":
    main()
