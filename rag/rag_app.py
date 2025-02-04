from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
import os
from langchain.prompts import PromptTemplate


class RAGApplication:
    def __init__(self, model_name="gpt-3.5-turbo-instruct", openai_api_key=None):
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )

        # Initialize LLM
        if openai_api_key is None:
            openai_api_key = ""
            if openai_api_key is None:
                raise ValueError(
                    "OpenAI API key must be provided or set as OPENAI_API_KEY environment variable"
                )

        self.llm = OpenAI(model_name=model_name, openai_api_key=openai_api_key)

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )

        self.vector_store = None

    def load_pdf(self, pdf_path):
        """Load and process a PDF file"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Split documents
        texts = self.text_splitter.split_documents(documents)

        # Create vector store
        self.vector_store = FAISS.from_documents(texts, self.embeddings)

        # Save vector store locally
        self.vector_store.save_local("faiss_index")

        return len(texts)

    def load_existing_vector_store(self):
        """Load an existing vector store"""
        if os.path.exists("faiss_index"):
            self.vector_store = FAISS.load_local("faiss_index", self.embeddings)
            return True
        return False

    def query(self, question, prompt_template=None):
        """Query the RAG system"""
        if not self.vector_store:
            raise ValueError(
                "No vector store loaded. Please load a PDF first or load an existing vector store."
            )

        if prompt_template is None:
            template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know.

Context: {context}

Question: {question}

Answer: """

            prompt_template = PromptTemplate(
                template=template, input_variables=["context", "question"]
            )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            chain_type_kwargs={"prompt": prompt_template},
        )

        return qa_chain.run(question)
