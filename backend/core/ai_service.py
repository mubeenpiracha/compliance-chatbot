# backend/core/ai_service.py
import os
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.vector_stores import VectorStoreQuery, MetadataFilters, ExactMatchFilter
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI

from backend.core.graph_agent import agent, AgentState
from langchain_core.messages import HumanMessage, AIMessage

from pinecone import Pinecone
from backend.core.config import PINECONE_API_KEY, OPENAI_API_KEY

# Enhanced retrieval imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer, CrossEncoder
    ENHANCED_RETRIEVAL = True
    print("Enhanced retrieval mode: ENABLED")
except ImportError:
    ENHANCED_RETRIEVAL = False
    print("Enhanced retrieval mode: DISABLED (install sentence-transformers and scikit-learn)")

from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

class RetrievalMode(Enum):
    VECTOR_ONLY = "vector_only" 
    HYBRID = "hybrid"
    ENHANCED = "enhanced"

@dataclass
class RetrievalConfig:
    mode: RetrievalMode = RetrievalMode.ENHANCED if ENHANCED_RETRIEVAL else RetrievalMode.VECTOR_ONLY
    top_k_initial: int = 20
    top_k_final: int = 5
    enable_reranking: bool = ENHANCED_RETRIEVAL
    enable_query_expansion: bool = True

# Global variables
index = None
enhanced_retriever = None
retrieval_config = RetrievalConfig()

class EnhancedAIRetriever:
    """Enhanced retrieval system integrated with existing AI service."""
    
    def __init__(self, vector_index, config: RetrievalConfig):
        self.index = vector_index
        self.config = config
        
        # Initialize models if enhanced mode is available
        self.reranker = None
        self.tfidf_vectorizer = None
        self.document_cache = {}
        
        if ENHANCED_RETRIEVAL and config.enable_reranking:
            try:
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-2-v2')
                print("  ✓ Loaded cross-encoder for reranking")
            except:
                print("  Warning: Could not load reranker")
                self.config.enable_reranking = False
    
    def expand_query(self, query: str) -> str:
        """Expand query with regulatory synonyms."""
        if not self.config.enable_query_expansion:
            return query
        
        regulatory_expansions = {
            'requirement': ['obligation', 'mandate', 'rule'],
            'compliance': ['adherence', 'conformity'], 
            'violation': ['breach', 'infringement'],
            'penalty': ['fine', 'sanction'],
            'license': ['permit', 'authorization'],
            'capital': ['funds', 'reserves'],
            'risk': ['exposure', 'hazard'],
            'audit': ['examination', 'inspection']
        }
        
        words = query.lower().split()
        expanded_terms = []
        
        for word in words:
            if word in regulatory_expansions:
                expanded_terms.extend(regulatory_expansions[word][:2])
        
        if expanded_terms:
            return f"{query} {' '.join(expanded_terms[:3])}"
        return query
    
    def enhanced_retrieve(self, query: str, jurisdiction: str = None) -> List:
        """Enhanced retrieval with multiple strategies."""
        print(f"Enhanced retrieval for: '{query}' (mode: {self.config.mode})")
        
        if self.config.mode == RetrievalMode.VECTOR_ONLY:
            return self._vector_only_retrieve(query, jurisdiction)
        elif self.config.mode == RetrievalMode.ENHANCED and ENHANCED_RETRIEVAL:
            return self._enhanced_retrieve(query, jurisdiction)
        else:
            return self._vector_only_retrieve(query, jurisdiction)
    
    def _vector_only_retrieve(self, query: str, jurisdiction: str = None) -> List:
        """Standard vector retrieval."""
        try:
            query_engine = self.index.as_query_engine(
                similarity_top_k=self.config.top_k_final,
                filters=self._build_filters(jurisdiction) if jurisdiction else None
            )
            response = query_engine.query(query)
            
            retrieved_docs = []
            for node in getattr(response, "source_nodes", []):
                content = getattr(node.node, "get_content", lambda: None)()
                metadata = getattr(node.node, "metadata", {})
                if content:
                    retrieved_docs.append({
                        'content': content,
                        'metadata': metadata,
                        'score': getattr(node, 'score', 0.0)
                    })
            
            return retrieved_docs
        except Exception as e:
            print(f"Vector retrieval failed: {e}")
            return []
    
    def _enhanced_retrieve(self, query: str, jurisdiction: str = None) -> List:
        """Enhanced retrieval with query expansion and reranking."""
        try:
            # Query expansion
            expanded_query = self.expand_query(query)
            
            # Multiple retrieval strategies
            results = {}
            
            # Original query
            original_results = self._vector_only_retrieve(query, jurisdiction)
            results['original'] = original_results
            
            # Expanded query
            if expanded_query != query:
                expanded_results = self._vector_only_retrieve(expanded_query, jurisdiction)
                results['expanded'] = expanded_results
            
            # Combine results using reciprocal rank fusion
            if len(results) > 1:
                fused_results = self._reciprocal_rank_fusion(results)
            else:
                fused_results = original_results
            
            # Reranking
            if self.config.enable_reranking and self.reranker and fused_results:
                reranked_results = self._rerank_documents(query, fused_results)
                return reranked_results[:self.config.top_k_final]
            
            return fused_results[:self.config.top_k_final]
            
        except Exception as e:
            print(f"Enhanced retrieval failed: {e}")
            return self._vector_only_retrieve(query, jurisdiction)
    
    def _reciprocal_rank_fusion(self, result_sets: Dict[str, List], k: int = 60) -> List:
        """Combine multiple result sets using RRF."""
        doc_scores = {}
        
        for strategy_name, results in result_sets.items():
            for rank, doc in enumerate(results):
                doc_key = hash(doc.get('content', '')[:100])  # Simple doc identification
                rrf_score = 1 / (k + rank + 1)
                
                if doc_key in doc_scores:
                    doc_scores[doc_key]['score'] += rrf_score
                else:
                    doc_scores[doc_key] = {
                        'document': doc,
                        'score': rrf_score
                    }
        
        # Sort by RRF score
        fused_results = [data['document'] for data in sorted(doc_scores.values(), key=lambda x: x['score'], reverse=True)]
        print(f"  RRF fusion: {sum(len(r) for r in result_sets.values())} -> {len(fused_results)}")
        return fused_results
    
    def _rerank_documents(self, query: str, documents: List[Dict]) -> List[Dict]:
        """Rerank documents using cross-encoder."""
        if not self.reranker or not documents:
            return documents
        
        try:
            # Prepare pairs for cross-encoder
            pairs = []
            for doc in documents:
                content = doc.get('content', '')[:512]  # Limit for efficiency
                pairs.append([query, content])
            
            # Get reranking scores
            scores = self.reranker.predict(pairs)
            
            # Update documents with reranking scores
            for doc, score in zip(documents, scores):
                doc['rerank_score'] = float(score)
            
            # Sort by reranking score
            reranked = sorted(documents, key=lambda x: x.get('rerank_score', 0), reverse=True)
            print(f"  Reranked {len(reranked)} documents")
            return reranked
            
        except Exception as e:
            print(f"Reranking failed: {e}")
            return documents
    
    def _build_filters(self, jurisdiction: str) -> MetadataFilters:
        """Build metadata filters for jurisdiction."""
        if not jurisdiction:
            return None
        
        return MetadataFilters(filters=[
            ExactMatchFilter(key="jurisdiction", value=jurisdiction.upper())
        ])

def initialize_ai_service():
    """Initialize AI service with enhanced retrieval capabilities."""
    global index, enhanced_retriever
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index("compliance-bot-index")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    index = VectorStoreIndex.from_vector_store(vector_store)
    
    # Initialize enhanced retriever
    enhanced_retriever = EnhancedAIRetriever(index, retrieval_config)
    
    print(f"AI Service Initialized (Enhanced retrieval: {ENHANCED_RETRIEVAL})")

def get_enhanced_documents(query: str, jurisdiction: str = None) -> List:
    """Get documents using enhanced retrieval."""
    if not enhanced_retriever:
        print("Enhanced retriever not initialized, falling back to basic retrieval")
        return []
    
    return enhanced_retriever.enhanced_retrieve(query, jurisdiction)

def get_ai_response(user_message: str, history: list, jurisdiction: str) -> dict:
    """
    Gets a response from the AI agent.

    This function now invokes the new LangGraph-based agent. It prepares the
    initial state and then calls the compiled graph.
    """
    if index is None:
        return {"answer": "Error: AI Service Index is not initialized.", "sources": []}

    print(f"--- Invoking Graph Agent for jurisdiction: {jurisdiction} ---")

    # 1. Convert the incoming history (a list of dicts) to a list of BaseMessages
    chat_history = []
    for msg in history:
        if msg['sender'] == 'user':
            chat_history.append(HumanMessage(content=msg['text']))
        else:
            chat_history.append(AIMessage(content=msg['text']))

    # 2. Prepare the initial state for the graph
    initial_state: AgentState = {
        "original_query": user_message,
        "jurisdiction": jurisdiction,
        "chat_history": chat_history,
        # The rest of the fields will be populated by the graph nodes
        "contextualized_query": "",
        "query_confidence": 0.0,
        "query_type": "COMPLIANCE",
        "retrieved_docs": [],
        "conversational_response": "",
        "compliance_response": "",
        "final_response": "",
        "final_response_with_sources": {},
        "should_retrieve": True,
        "should_converse": True,
        "exploration_suggestions": []
    }

    # 3. Invoke the agent graph
    # The `agent.stream()` method can be used for streaming, but for now,
    # we'll use `invoke` to get the final state.
    try:
        final_state = agent.invoke(initial_state)

        # 4. Determine the response based on the final state
        if final_state.get("final_response_with_sources"):
            return final_state["final_response_with_sources"]
        elif final_state.get("final_response"):
            return {"answer": final_state["final_response"], "sources": final_state.get("retrieved_docs", [])}
        else:
            return {"answer": "I'm sorry, I encountered an issue and couldn't process your request.", "sources": []}

    except Exception as e:
        print(f"An error occurred during graph execution: {e}")
        # Provide a more detailed error for debugging if possible
        import traceback
        traceback.print_exc()
        return {"answer": "Sorry, I encountered a critical error while processing your request.", "sources": []}

initialize_ai_service()