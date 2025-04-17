from flask import Flask, render_template, request, jsonify
from rag_setup import RAGSystem
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize RAG system
try:
    rag = RAGSystem()
    # Try to load existing vector store
    if not rag.load_vector_store():
        logger.warning("No existing vector store found. Please run rag_setup.py first to process documents.")
except Exception as e:
    logger.error(f"Error initializing RAG system: {e}")
    rag = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    if not rag:
        return jsonify({'error': 'RAG system not initialized. Please run rag_setup.py first.'}), 500
        
    try:
        data = request.get_json()
        query_text = data.get('query', '').strip()
        
        if not query_text:
            return jsonify({'error': 'Please enter a query'}), 400
            
        # Get results from RAG system
        results = rag.query(query_text, k=3)  # Get top 3 results
        
        # Format results
        formatted_results = []
        for i, doc in enumerate(results, 1):
            formatted_results.append({
                'id': i,
                'content': doc.page_content,
                'metadata': doc.metadata
            })
            
        return jsonify({
            'results': formatted_results
        })
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 