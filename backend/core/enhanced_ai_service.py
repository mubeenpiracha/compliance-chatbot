# Enhanced AI service with multi-namespace support and content retrieval
# backend/core/enhanced_ai_service.py

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass
from langchain_core.documents import Document
from pinecone import Pinecone
from llama_index.embeddings.openai import OpenAIEmbedding
from backend.core.config import PINECONE_API_KEY, OPENAI_API_KEY

@dataclass
class QueryIntent:
    """Structured representation of query analysis"""
    topics: List[str]
    jurisdiction: Optional[str] 
    document_types: List[str]
    specificity: str  # "high", "medium", "low"
    confidence: float
    predicted_namespaces: List[str]

@dataclass 
class SearchResult:
    """Enhanced search result with content loading"""
    content: str
    metadata: Dict[str, Any]
    score: float
    namespace: str
    content_loaded: bool = False

class SmartNamespaceManager:
    """Manages namespace operations and content retrieval"""
    
    def __init__(self, pinecone_client: Pinecone, index_name: str):
        self.pc = pinecone_client
        self.index = self.pc.Index(index_name)
        self.content_store = Path("./content_store")
        self._namespace_cache = None
        self._jurisdiction_map = None
        
    def get_all_namespaces(self) -> List[str]:
        """Get all available namespaces with caching"""
        if self._namespace_cache is None:
            try:
                stats = self.index.describe_index_stats()
                if hasattr(stats, 'namespaces') and stats.namespaces:
                    self._namespace_cache = list(stats.namespaces.keys())
                else:
                    self._namespace_cache = [""]  # Default namespace
            except Exception as e:
                print(f"Error getting namespaces: {e}")
                self._namespace_cache = [""]
        return self._namespace_cache
    
    def get_namespaces_by_jurisdiction(self, jurisdiction: str) -> List[str]:
        """Filter namespaces by jurisdiction"""
        all_namespaces = self.get_all_namespaces()
        
        if not jurisdiction:
            return all_namespaces
        
        # Build jurisdiction map if not cached
        if self._jurisdiction_map is None:
            self._build_jurisdiction_map()
        
        return self._jurisdiction_map.get(jurisdiction.upper(), all_namespaces)
    
    def _build_jurisdiction_map(self):
        """Build mapping of jurisdictions to namespaces"""
        self._jurisdiction_map = {"DIFC": [], "ADGM": [], "COMMON": []}
        
        for namespace in self.get_all_namespaces():
            if any(difc_indicator in namespace.lower() for difc_indicator in ["difc", "dfsa"]):
                self._jurisdiction_map["DIFC"].append(namespace)
            elif any(adgm_indicator in namespace.lower() for adgm_indicator in ["adgm", "fsra"]):
                self._jurisdiction_map["ADGM"].append(namespace)
            else:
                self._jurisdiction_map["COMMON"].append(namespace)
    
    def load_content_from_uri(self, content_uri: str) -> Optional[str]:
        """Load content from local storage URI"""
        try:
            if not content_uri:
                return None
                
            # Handle both absolute and relative paths
            if content_uri.startswith("content_store/"):
                file_path = Path(content_uri)
            else:
                file_path = Path(content_uri)
                
            if file_path.exists():
                return file_path.read_text(encoding="utf-8")
            else:
                print(f"Content file not found: {file_path}")
                return None
        except Exception as e:
            print(f"Error loading content from {content_uri}: {e}")
            return None

class QueryAnalyzer:
    """Analyzes queries to determine search strategy"""
    
    def __init__(self):
        # Define topic-to-namespace patterns
        self.topic_patterns = {
            "capital": ["prudential", "capital", "investment"],
            "aml": ["aml", "anti-money", "sanctions"],
            "employment": ["employment", "hr", "staff"],
            "licensing": ["licensing", "authorization", "permit"],
            "conduct": ["conduct", "business", "ethics"],
            "fund": ["fund", "collective", "investment-scheme"],
            "banking": ["banking", "deposit", "credit"],
            "insurance": ["insurance", "takaful", "actuarial"],
        }
        
        self.jurisdiction_indicators = {
            "DIFC": ["difc", "dfsa", "dubai international", "dubai financial"],
            "ADGM": ["adgm", "fsra", "abu dhabi global", "abu dhabi financial"],
        }
    
    def analyze_query(self, query: str, jurisdiction: str = None) -> QueryIntent:
        """Analyze query to determine search strategy"""
        query_lower = query.lower()
        
        # Extract topics
        topics = self._extract_topics(query_lower)
        
        # Determine jurisdiction
        detected_jurisdiction = self._detect_jurisdiction(query_lower) or jurisdiction
        
        # Predict document types
        document_types = self._predict_document_types(query_lower, topics)
        
        # Assess specificity
        specificity = self._assess_specificity(query_lower, topics)
        
        # Generate predicted namespaces
        predicted_namespaces = self._predict_namespaces(topics, document_types)
        
        # Calculate confidence
        confidence = self._calculate_confidence(query_lower, topics, detected_jurisdiction)
        
        return QueryIntent(
            topics=topics,
            jurisdiction=detected_jurisdiction,
            document_types=document_types,
            specificity=specificity,
            confidence=confidence,
            predicted_namespaces=predicted_namespaces
        )
    
    def _extract_topics(self, query: str) -> List[str]:
        """Extract relevant compliance topics from query"""
        topics = []
        for topic, keywords in self.topic_patterns.items():
            if any(keyword in query for keyword in keywords):
                topics.append(topic)
        
        # Add specific regulatory terms
        regulatory_terms = [
            "penalty", "fine", "violation", "breach", "requirement", 
            "obligation", "procedure", "application", "approval", "notification"
        ]
        for term in regulatory_terms:
            if term in query:
                topics.append(term)
        
        return list(set(topics))
    
    def _detect_jurisdiction(self, query: str) -> Optional[str]:
        """Detect jurisdiction from query"""
        for jurisdiction, indicators in self.jurisdiction_indicators.items():
            if any(indicator in query for indicator in indicators):
                return jurisdiction
        return None
    
    def _predict_document_types(self, query: str, topics: List[str]) -> List[str]:
        """Predict likely document types"""
        doc_types = []
        
        type_indicators = {
            "rulebook": ["rule", "regulation", "requirement"],
            "guidance": ["guidance", "guide", "how to", "procedure"],
            "form": ["form", "application", "template"],
            "circular": ["circular", "notice", "announcement"],
        }
        
        for doc_type, indicators in type_indicators.items():
            if any(indicator in query for indicator in indicators):
                doc_types.append(doc_type)
                
        return doc_types
    
    def _assess_specificity(self, query: str, topics: List[str]) -> str:
        """Assess how specific the query is"""
        specific_indicators = ["section", "article", "paragraph", "rule", "form number"]
        general_indicators = ["overview", "general", "about", "explain", "what is"]
        
        if any(indicator in query for indicator in specific_indicators):
            return "high"
        elif any(indicator in query for indicator in general_indicators) or len(topics) == 0:
            return "low"
        else:
            return "medium"
    
    def _predict_namespaces(self, topics: List[str], document_types: List[str]) -> List[str]:
        """Predict likely namespaces based on analysis"""
        predicted = []
        
        # Map topics to likely namespace patterns
        for topic in topics:
            if topic == "aml":
                predicted.extend(["anti-money-laundering", "aml", "sanctions"])
            elif topic == "capital":
                predicted.extend(["prudential", "capital-requirements", "investment"])
            elif topic == "employment":
                predicted.extend(["employment", "hr"])
            elif topic == "fund":
                predicted.extend(["fund", "collective-investment"])
            # Add more mappings as needed
        
        return list(set(predicted))
    
    def _calculate_confidence(self, query: str, topics: List[str], jurisdiction: Optional[str]) -> float:
        """Calculate confidence in query analysis"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence for clear topics
        if topics:
            confidence += 0.2 * min(len(topics), 2)  # Cap at 2 topics
            
        # Boost for jurisdiction specificity
        if jurisdiction:
            confidence += 0.2
            
        # Boost for regulatory language
        regulatory_language = ["shall", "must", "required", "prohibited", "permitted"]
        if any(term in query for term in regulatory_language):
            confidence += 0.1
            
        return min(confidence, 1.0)

class EnhancedRetriever:
    """Enhanced retriever with multi-namespace support"""
    
    def __init__(self, embedder: OpenAIEmbedding, namespace_manager: SmartNamespaceManager):
        self.embedder = embedder
        self.namespace_manager = namespace_manager
        self.query_analyzer = QueryAnalyzer()
        
    def smart_retrieve(self, query: str, jurisdiction: str = None, top_k: int = 10) -> List[SearchResult]:
        """Smart retrieval with multi-namespace search"""
        print(f"🔍 Smart retrieval for: '{query}'")
        
        # Analyze query
        intent = self.query_analyzer.analyze_query(query, jurisdiction)
        print(f"  📊 Intent: topics={intent.topics}, specificity={intent.specificity}, confidence={intent.confidence:.2f}")
        
        # Determine search strategy
        namespaces_to_search = self._select_namespaces(intent, jurisdiction)
        print(f"  🎯 Searching {len(namespaces_to_search)} namespaces")
        
        # Execute search
        all_results = self._search_multiple_namespaces(query, namespaces_to_search, top_k)
        
        # Load content for results
        enhanced_results = []
        for result in all_results[:top_k]:
            content = self._load_result_content(result)
            enhanced_results.append(SearchResult(
                content=content,
                metadata=result.get('metadata', {}),
                score=result.get('score', 0.0),
                namespace=result.get('namespace', ''),
                content_loaded=bool(content)
            ))
        
        print(f"  ✅ Retrieved {len(enhanced_results)} results with content loaded")
        return enhanced_results
    
    def _select_namespaces(self, intent: QueryIntent, jurisdiction: str) -> List[str]:
        """Select namespaces to search based on intent"""
        if intent.specificity == "high" and intent.predicted_namespaces:
            # High specificity: use predicted namespaces
            all_namespaces = self.namespace_manager.get_all_namespaces()
            matched_namespaces = []
            for predicted in intent.predicted_namespaces:
                matched = [ns for ns in all_namespaces if predicted in ns.lower()]
                matched_namespaces.extend(matched)
            
            if matched_namespaces:
                return list(set(matched_namespaces))
        
        # Medium/Low specificity or no predictions: use jurisdiction filtering
        target_jurisdiction = intent.jurisdiction or jurisdiction
        if target_jurisdiction:
            return self.namespace_manager.get_namespaces_by_jurisdiction(target_jurisdiction)
        
        # Fallback: all namespaces
        return self.namespace_manager.get_all_namespaces()
    
    def _search_multiple_namespaces(self, query: str, namespaces: List[str], top_k: int) -> List[Dict]:
        """Search across multiple namespaces and aggregate results"""
        # Generate query embedding
        query_vector = self.embedder.get_text_embedding(query)
        
        all_results = []
        results_per_namespace = max(2, top_k // max(len(namespaces), 1))
        
        for namespace in namespaces:
            try:
                # Search this namespace
                response = self.namespace_manager.index.query(
                    vector=query_vector,
                    namespace=namespace,
                    top_k=results_per_namespace,
                    include_metadata=True,
                    include_values=False
                )
                
                # Process results
                for match in response.matches:
                    result = {
                        'id': match.id,
                        'score': match.score,
                        'metadata': match.metadata,
                        'namespace': namespace
                    }
                    all_results.append(result)
                    
            except Exception as e:
                print(f"  ⚠️ Error searching namespace '{namespace}': {e}")
                continue
        
        # Sort by score and return top results
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return all_results[:top_k]
    
    def _load_result_content(self, result: Dict) -> str:
        """Load content for a search result"""
        metadata = result.get('metadata', {})
        
        # Try to load from content_uri
        content_uri = metadata.get('content_uri')
        if content_uri:
            content = self.namespace_manager.load_content_from_uri(content_uri)
            if content:
                return content
        
        # Fallback: try to extract from metadata (legacy support)
        if '_node_content' in metadata:
            try:
                import json
                node_data = json.loads(metadata['_node_content'])
                if isinstance(node_data, dict) and 'text' in node_data:
                    return node_data['text']
            except:
                pass
        
        # Last resort: return any text-like field
        for field in ['content', 'text', 'page_content']:
            if field in metadata:
                return str(metadata[field])
        
        return "Content not available"

# Main enhanced AI service class
class EnhancedAIService:
    """Enhanced AI service with smart namespace management"""
    
    def __init__(self):
        self.embedder = OpenAIEmbedding(
            model="text-embedding-3-large",
            api_key=OPENAI_API_KEY,
            dimensions=1536,
        )
        
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.namespace_manager = SmartNamespaceManager(self.pc, "compliance-bot-index")
        self.retriever = EnhancedRetriever(self.embedder, self.namespace_manager)
        
        print("✅ Enhanced AI Service initialized")
    
    def get_enhanced_documents(self, query: str, jurisdiction: str = None, top_k: int = 10) -> List[Dict]:
        """Get documents using enhanced multi-namespace retrieval"""
        try:
            results = self.retriever.smart_retrieve(query, jurisdiction, top_k)
            
            # Convert to expected format for compatibility
            documents = []
            for result in results:
                if result.content_loaded:
                    documents.append({
                        'content': result.content,
                        'metadata': result.metadata,
                        'score': result.score,
                        'namespace': result.namespace
                    })
            
            return documents
            
        except Exception as e:
            print(f"❌ Enhanced retrieval failed: {e}")
            return []
    
    def refresh_namespace_cache(self):
        """Refresh the namespace cache"""
        self.namespace_manager._namespace_cache = None
        self.namespace_manager._jurisdiction_map = None
        print("🔄 Namespace cache refreshed")

# Global instance for compatibility
enhanced_ai_service = None

def initialize_enhanced_ai_service():
    """Initialize the enhanced AI service"""
    global enhanced_ai_service
    enhanced_ai_service = EnhancedAIService()
    return enhanced_ai_service

def get_enhanced_documents(query: str, jurisdiction: str = None, top_k: int = 10) -> List[Dict]:
    """Compatibility function for existing code"""
    if enhanced_ai_service is None:
        initialize_enhanced_ai_service()
    
    return enhanced_ai_service.get_enhanced_documents(query, jurisdiction, top_k)

# Enhanced wrapper for graph agent integration
def get_ai_response(user_message: str, history: list, jurisdiction: str) -> dict:
    """Enhanced AI response using the graph agent with smart namespace retrieval"""
    try:
        # Import the graph agent - it should use the enhanced service
        from backend.core.graph_agent import agent, AgentState
        from langchain_core.messages import HumanMessage, AIMessage
        
        # Initialize enhanced service 
        if enhanced_ai_service is None:
            initialize_enhanced_ai_service()
        
        # Convert history to langchain messages
        chat_history = []
        for msg in history:
            if msg.get('sender') == 'user':
                chat_history.append(HumanMessage(content=msg['text']))
            elif msg.get('sender') == 'ai':
                chat_history.append(AIMessage(content=msg['text']))
        
        # Prepare initial state
        initial_state = AgentState(
            original_query=user_message,
            jurisdiction=jurisdiction,
            chat_history=chat_history,
            contextualized_query="",
            query_confidence=0.0,
            query_type="COMPLIANCE",
            regulatory_topics=[],
            jurisdiction_indicators=[],
            document_types=[],
            retrieved_docs=[],
            conversational_response="",
            compliance_response="",
            final_response="",
            final_response_with_sources={},
            should_retrieve=False,
            should_converse=True,
            exploration_suggestions=[]
        )
        
        # Run the agent
        result = agent.invoke(initial_state)
        
        # Extract sources from retrieved documents
        sources = []
        if result.get('retrieved_docs'):
            for doc in result['retrieved_docs']:
                if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                    sources.append({
                        'page_content': doc.page_content,
                        'metadata': doc.metadata
                    })
        
        return {
            'answer': result.get('final_response', 'I apologize, but I encountered an error processing your request.'),
            'sources': sources
        }
        
    except Exception as e:
        print(f"❌ Enhanced AI response failed: {e}")
        return {
            'answer': 'I apologize, but I encountered an error processing your request. Please try again.',
            'sources': []
        }
