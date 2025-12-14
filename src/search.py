import numpy as np
from src.embedding import EmbeddingPipeline
from src.vectorstore import CHROMAVectorStore
from typing import List, Dict, Any, Tuple, Optional
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os

class RAGRetriever:
    """
    Improved RAG Retriever with:
    - Similarity threshold filtering
    - Conversation context awareness
    - Better prompt engineering
    """
    
    def __init__(self, vector_store: CHROMAVectorStore, llm_repo: str = 'deepseek-ai/DeepSeek-R1'):
        self.embedding_manager = EmbeddingPipeline()
        self.vector_store = vector_store
        load_dotenv()
        token = os.getenv('HF_API_TOKEN')
        endpoint = HuggingFaceEndpoint(
            repo_id=llm_repo, 
            huggingfacehub_api_token=token,
            max_new_tokens=1024,
            temperature=0.3  # Lower for more focused answers
        )
        self.llm = ChatHuggingFace(llm=endpoint)
        
        # Configurable thresholds
        self.min_similarity_threshold = 0.25  # Filter out low-quality matches
        self.top_k = 10  # Retrieve more for better recall

    def search(
        self, 
        query: str, 
        top_k: int = None, 
        filter: dict = None,
        conversation_context: str = ""
    ) -> str:
        """
        Main search method with improved accuracy
        """
        # Handle simple greetings naturally without RAG
        greeting_words = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
        if query.lower().strip() in greeting_words or query.lower().strip().rstrip('!') in greeting_words:
            return "Hello! How can I help you with the document today?"
        
        top_k = top_k or self.top_k
        results = self.retrieve(query, top_k, filter)
        
        # Filter by similarity threshold
        results = [
            doc for doc in results 
            if doc.get('similarity_score', 0) >= self.min_similarity_threshold
        ]
        
        if not results:
            return "I couldn't find relevant information in the uploaded documents. Please try rephrasing or make sure you've uploaded a relevant document."
        
        # Build context from retrieved documents
        context = "\n\n---\n\n".join([
            f"[Source: {doc['metadata'].get('source_pdf', 'Unknown')}, Page {doc['metadata'].get('page', 'N/A')}]\n{doc['content']}" 
            for doc in results
        ])
        
        # Build improved prompt with conversation context
        prompt = self._build_prompt(query, context, conversation_context)
        
        try:
            response = self.llm.invoke(prompt)
            # Clean response - strip DeepSeek's thinking tags
            clean_response = self._clean_response(response.content)
            return clean_response
        except Exception as e:
            print(f"LLM Error: {e}")
            return f"Error generating response: {str(e)}"
    
    def _clean_response(self, text: str) -> str:
        """
        Clean the LLM response by removing thinking tags and internal reasoning
        """
        import re
        
        # Remove <think>...</think> tags (DeepSeek-R1 thinking)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # Remove any remaining thinking patterns
        lines = text.split('\n')
        clean_lines = []
        skip_mode = False
        
        for line in lines:
            # Skip lines that look like internal reasoning
            lower_line = line.lower().strip()
            if any(phrase in lower_line for phrase in [
                'okay, let me', 'hmm,', 'let me think', 'i should', 
                "the user is", "they're probably", "best to", "i'll respond",
                "looking at", "this is clearly"
            ]):
                continue
            clean_lines.append(line)
        
        result = '\n'.join(clean_lines).strip()
        
        # If result is empty after cleaning, return a safe fallback
        if not result:
            return "I found relevant information but had trouble formatting the response. Please try asking your question again."
        
        return result
    
    def _build_prompt(self, query: str, context: str, conversation_context: str = "") -> str:
        """
        Build an improved prompt for better accuracy
        """
        conversation_section = ""
        if conversation_context:
            conversation_section = f"""
Previous Conversation:
{conversation_context}
---
"""
        
        prompt = f"""You are a helpful document assistant. Answer the user's question based on the document context provided.

RULES:
- Be concise and direct - no lengthy explanations unless needed
- Only use information from the context below
- If the answer isn't in the documents, say "I don't have that information in the documents"
- Do NOT show your reasoning or thinking process
- Just give the answer directly

{conversation_section}
CONTEXT:
{context}

---

QUESTION: {query}

ANSWER:"""
        
        return prompt
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 10, 
        filter: dict = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents with improved scoring
        """
        print(f"Retrieving documents for query: '{query}' with filter: {filter}")
        
        # DON'T preprocess the query - preserve user's original intent
        # Only preprocess for embedding (to match preprocessed documents)
        query_for_embedding = self.embedding_manager.preprocess_text(query)
        query_embedding = self.embedding_manager.embed_texts([query_for_embedding])[0]
        
        try:
            # Prepare query arguments
            query_args = {
                "query_embeddings": [query_embedding],
                "n_results": top_k
            }
            if filter:
                query_args["where"] = filter

            results = self.vector_store.collection.query(**query_args)
            retrieved_docs = []
            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                ids = results['ids'][0]
                distances = results['distances'][0]

                for i, (doc, metadata, id, distance) in enumerate(zip(documents, metadatas, ids, distances)):
                    # ChromaDB uses L2 distance by default
                    # Convert to similarity score (1 / (1 + distance))
                    similarity_score = 1 / (1 + distance)

                    retrieved_docs.append({
                        'id': id,
                        'content': doc,
                        'metadata': metadata,
                        'similarity_score': similarity_score,
                        'distance': distance,
                        'rank': i + 1
                    })
                
                print(f"Retrieved {len(retrieved_docs)} documents")
                print(f"Score range: {retrieved_docs[0]['similarity_score']:.3f} - {retrieved_docs[-1]['similarity_score']:.3f}")
            else:
                print("No documents found")

            return retrieved_docs
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            import traceback
            traceback.print_exc()
            return []
