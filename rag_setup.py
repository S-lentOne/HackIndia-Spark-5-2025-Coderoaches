from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, JSONLoader
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.documents import Document
from datasets import load_dataset
import os
import sys
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
import json
from tqdm import tqdm
import torch
import gc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class RAGSystem:
    def __init__(self, 
                 documents_dir="/home/Mika/Documents/HackIndia-Spark-5-2025-Coderoaches/DATA", 
                 embedding_model="sentence-transformers/all-mpnet-base-v2",
                 vector_store_path="vector_store"):
        self.documents_dir = documents_dir
        self.embedding_model = embedding_model
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_store = None
        self.vector_store_path = vector_store_path
        
    def save_vector_store(self):
        """Save the vector store to disk"""
        if self.vector_store:
            self.vector_store.save_local(self.vector_store_path)
            print(f"\nVector store saved to {self.vector_store_path}")
            
    def load_vector_store(self):
        """Load the vector store from disk if it exists"""
        if os.path.exists(self.vector_store_path):
            try:
                self.vector_store = FAISS.load_local(
                    self.vector_store_path,
                    self.embeddings
                )
                print(f"\nLoaded existing vector store from {self.vector_store_path}")
                return True
            except Exception as e:
                print(f"\nError loading vector store: {e}")
                return False
        return False

    def load_json_documents(self, batch_size: int = 25):
        """Load documents from Hugging Face dataset in batches"""
        documents = []
        dataset = load_dataset("nbroad/small_arxiv_classification")
        
        # Process only the train split
        train_data = dataset['train']
        print(f"\nProcessing {len(train_data)} documents from the dataset...")
        
        for i in tqdm(range(len(train_data)), desc="Loading documents"):
            try:
                item = train_data[i]
                # Extract text field (contains the content)
                content = item['text']
                
                if content and content.strip():
                    doc = Document(
                        page_content=content,
                        metadata={
                            'label': str(item['label']),
                            'id': str(i)
                        }
                    )
                    documents.append(doc)
                    
                    # Process in batches
                    if len(documents) >= batch_size:
                        yield documents
                        documents = []
                        
                        # Clear CUDA cache
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        
            except Exception as e:
                logger.warning(f"Error processing item {i}: {e}")
                continue
                
        # Yield any remaining documents
        if documents:
            yield documents

    def process_documents(self, documents, chunk_size=250, chunk_overlap=25):
        """Split documents into chunks and create embeddings"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        texts = text_splitter.split_documents(documents)
        return texts

    def initialize_vector_store(self, texts):
        """Initialize the vector store with processed documents"""
        self.vector_store = FAISS.from_documents(
            documents=texts,
            embedding=self.embeddings
        )
        return self.vector_store

    def add_to_vector_store(self, texts):
        """Add documents to an existing vector store"""
        if not self.vector_store:
            return self.initialize_vector_store(texts)
        self.vector_store.add_documents(texts)
        
        # Memory cleanup after adding documents
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def query(self, question, k=4):
        """Query the vector store for relevant documents"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please load and process documents first.")
        
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(question)
        return docs

def main():
    # Initialize RAG system
    rag = RAGSystem()
    
    # Try to load existing vector store
    if rag.load_vector_store():
        print("Using existing vector store. Skipping document processing.")
    else:
        print("No existing vector store found. Processing documents...")
        total_docs = 0
        total_chunks = 0
        
        try:
            # Process documents in batches
            for batch in rag.load_json_documents(batch_size=25):
                if batch:
                    print(f"\nProcessing batch of {len(batch)} documents...")
                    texts = rag.process_documents(batch)
                    
                    if texts:
                        print(f"Adding {len(texts)} chunks to vector store...")
                        if not rag.vector_store:
                            rag.initialize_vector_store(texts)
                        else:
                            rag.add_to_vector_store(texts)
                        
                        total_docs += len(batch)
                        total_chunks += len(texts)
                        print(f"Progress: {total_docs} documents processed, {total_chunks} chunks created")
                    
                    # Memory cleanup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user. Saving progress...")
        except Exception as e:
            print(f"\nError during processing: {e}")
            raise
        
        print("\nVector store initialization completed!")
        print(f"Total documents processed: {total_docs}")
        print(f"Total chunks created: {total_chunks}")
        
        # Save the vector store
        rag.save_vector_store()
    
    # Query the vector store
    while True:
        try:
            question = input("\nEnter your question (or 'quit' to exit): ")
            if question.lower() == 'quit':
                break
                
            print(f"\nQuerying: {question}")
            results = rag.query(question)
            print("\nTop results:")
            for i, doc in enumerate(results, 1):
                print(f"\nResult {i}:")
                print(doc.page_content[:200] + "...")  # Print first 200 chars
        except Exception as e:
            print(f"Error during query: {e}")
            continue

if __name__ == "__main__":
    main() 