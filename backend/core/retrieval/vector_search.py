"""
Vector search implementation for semantic similarity retrieval.
"""
import numpy as np
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from ..models.retrieval_models import RetrievalQuery, RetrievedDocument, DocumentSource, DocumentType
import logging

logger = logging.getLogger(__name__)


class VectorSearchEngine:
    """Handles semantic similarity search using vector embeddings."""
    
    def __init__(self, client: AsyncOpenAI, vector_service):
        self.client = client
        self.vector_service = vector_service
        
    async def search(self, query: RetrievalQuery) -> List[RetrievedDocument]:
        """Perform vector similarity search."""
        
        try:
            # Generate query embedding
            query_embedding = await self._get_query_embedding(query.query_text)
            
            # Perform vector search
            search_results = await self.vector_service.search(
                query_vector=query_embedding,
                top_k=query.max_results,
                filter_params=self._build_filters(query)
            )
            
            # Convert to RetrievedDocument objects
            documents = []
            for result in search_results:
                if result['score'] >= query.min_relevance_score:
                    doc = await self._convert_to_retrieved_document(result, query)
                    documents.append(doc)
            
            logger.info(f"Vector search returned {len(documents)} documents for query: {query.query_text[:50]}...")
            return documents
            
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            return []
    
    async def _get_query_embedding(self, query_text: str) -> List[float]:
        """Generate embedding for the query text using OpenAI."""
        try:
            # Enhance query with regulatory context for better embedding
            enhanced_query = self._enhance_query_for_embedding(query_text)
            
            response = await self.client.embeddings.create(
                model="text-embedding-3-large",
                input=enhanced_query,
                dimensions=1536  # Match the Pinecone index dimension
            )
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            # Return a zero vector as fallback
            return [0.0] * 1536
    
    def _enhance_query_for_embedding(self, query_text: str) -> str:
        """Enhance query with regulatory context for better semantic matching."""
        
        # Detect if this is a definition query and enhance accordingly
        query_lower = query_text.lower()
        
        if any(term in query_lower for term in ['what is', 'define', 'definition', 'meaning']):
            if any(term in query_lower for term in ['fund', 'collective investment']):
                # For fund definition queries, add terms that appear in legal definitions
                return f"legal definition regulatory meaning: {query_text} arrangements with respect to property collective investment fund means"
            else:
                # For other definition queries, add general legal context
                return f"legal definition regulatory meaning: {query_text}"
        
        # Add regulatory context terms to improve semantic matching
        regulatory_context = "ADGM financial services regulation compliance"
        return f"{regulatory_context}: {query_text}"
    
    def _build_filters(self, query: RetrievalQuery) -> Dict[str, Any]:
        """Build metadata filters for vector search."""
        
        filters = {}
        
        if query.target_domains:
            # This is intentionally commented out to fix the bug where the 'domain'
            # metadata field does not exist in the ingested data. The new architecture
            # uses domains as context for the LLM, not as a hard filter.
            # filters['domain'] = query.target_domains
            pass
        
        if query.required_document_types:
            filters['document_type'] = [dt.value for dt in query.required_document_types]
        
        return filters
    
    async def _convert_to_retrieved_document(self, search_result: Dict, 
                                           query: RetrievalQuery) -> RetrievedDocument:
        """Convert vector search result to RetrievedDocument."""
        
        metadata = search_result.get('metadata', {})
        
        # Create document source
        source = DocumentSource(
            document_id=search_result['id'],
            document_type=DocumentType(metadata.get('document_type', 'guidance')),
            title=metadata.get('title', 'Unknown Document'),
            section=metadata.get('section'),
            subsection=metadata.get('subsection'), 
            authority_level=metadata.get('authority_level', 4),
            jurisdiction=metadata.get('jurisdiction', 'adgm')
        )
        
        # Generate highlights
        highlights = await self._generate_highlights(
            search_result['content'], query.query_text
        )
        
        return RetrievedDocument(
            source=source,
            content=search_result['content'],
            relevance_score=search_result['score'],
            retrieval_method="vector",
            match_highlights=highlights,
            context_window=search_result.get('context', search_result['content'])
        )
    
    async def _generate_highlights(self, content: str, query: str) -> List[str]:
        """Generate highlighted excerpts that match the query."""
        
        # Simple keyword highlighting - could be enhanced with more sophisticated matching
        query_terms = query.lower().split()
        content_lower = content.lower()
        
        highlights = []
        sentences = content.split('. ')
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(term in sentence_lower for term in query_terms):
                # Truncate if too long
                if len(sentence) > 200:
                    sentence = sentence[:200] + "..."
                highlights.append(sentence.strip())
                
        return highlights[:3]  # Return top 3 highlights
