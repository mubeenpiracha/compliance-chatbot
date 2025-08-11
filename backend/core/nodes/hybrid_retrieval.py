"""
Hybrid Retrieval Node - Orchestrates multiple retrieval strategies.
"""
import asyncio
from typing import Any, Dict, List
from openai import AsyncOpenAI
from .base_node import BaseNode
from ..models.query_models import ProcessingState
from ..models.retrieval_models import RetrievalQuery, DocumentType, HybridRetrievalResult
from ..retrieval.vector_search import VectorSearchEngine
from ..retrieval.keyword_search import KeywordSearchEngine
from ..retrieval.result_fusion import ResultFusion
import logging

logger = logging.getLogger(__name__)


class HybridRetrievalNode(BaseNode):
    """Orchestrates multiple retrieval strategies for comprehensive document search."""
    
    def __init__(self, client: AsyncOpenAI, vector_service, document_corpus):
        super().__init__("hybrid_retrieval")
        self.client = client
        self.vector_search = VectorSearchEngine(client, vector_service)
        self.keyword_search = KeywordSearchEngine(document_corpus)
        self.result_fusion = ResultFusion()
        
    async def process(self, state: ProcessingState, **kwargs) -> Dict[str, Any]:
        """Execute hybrid retrieval for all identified queries."""
        
        knowledge_gap = state.intermediate_results["knowledge_gap_identification"]["knowledge_gap"]
        context = state.intermediate_results["regulatory_context"]["context"]
        
        # Process all retrieval queries
        all_results = []
        for query_text in knowledge_gap.retrieval_queries:
            result = await self._execute_hybrid_search(query_text, context)
            all_results.append(result)
        
        # Consolidate results
        consolidated_results = self._consolidate_results(all_results)
        
        return {
            "retrieval_results": all_results,
            "consolidated_results": consolidated_results,
            "retrieval_summary": self._generate_summary(all_results)
        }
    
    async def _execute_hybrid_search(self, query_text: str, context) -> HybridRetrievalResult:
        """Execute all retrieval strategies for a single query."""
        
        # Create retrieval query
        retrieval_query = RetrievalQuery(
            query_text=query_text,
            query_type="fusion",  # Will use multiple types
            target_domains=context.regulatory_domains,
            required_document_types=self._get_relevant_document_types(context),
            max_results=15,  # Get more from each method for better fusion
            min_relevance_score=0.3
        )
        
        try:
            # Execute all retrieval strategies in parallel
            retrieval_tasks = {
                "vector": self._execute_vector_search(retrieval_query),
                "keyword": self._execute_keyword_search(retrieval_query),
                "regulatory": self._execute_regulatory_search(retrieval_query),
                "entity": self._execute_entity_search(retrieval_query)
            }
            
            # Wait for all searches to complete
            retrieval_results = {}
            for method, task in retrieval_tasks.items():
                try:
                    retrieval_results[method] = await task
                    logger.info(f"{method} search completed with {len(retrieval_results[method])} results")
                except Exception as e:
                    logger.error(f"{method} search failed: {str(e)}")
                    retrieval_results[method] = []
            
            # Fuse results
            hybrid_result = self.result_fusion.fuse_results(retrieval_results, retrieval_query)
            
            logger.info(f"Hybrid retrieval completed. Final documents: {len(hybrid_result.documents)}")
            return hybrid_result
            
        except Exception as e:
            logger.error(f"Hybrid retrieval failed for query '{query_text}': {str(e)}")
            # Return empty result
            return HybridRetrievalResult(
                query=retrieval_query,
                documents=[],
                retrieval_stats={"error": str(e)},
                coverage_assessment=0.0,
                fusion_confidence=0.0
            )
    
    async def _execute_vector_search(self, query: RetrievalQuery):
        """Execute vector similarity search."""
        return await self.vector_search.search(query)
    
    async def _execute_keyword_search(self, query: RetrievalQuery):
        """Execute keyword/BM25 search."""
        return await self.keyword_search.search(query)
    
    async def _execute_regulatory_search(self, query: RetrievalQuery):
        """Execute regulatory structure-aware search."""
        # For now, this is a placeholder - could implement document hierarchy navigation
        # This would search through regulatory document structure (sections, cross-references)
        
        # Enhanced query for regulatory structure
        regulatory_query = RetrievalQuery(
            query_text=f"regulatory definition requirement: {query.query_text}",
            query_type="regulatory",
            target_domains=query.target_domains,
            required_document_types=[DocumentType.LAW, DocumentType.REGULATION, DocumentType.RULEBOOK],
            max_results=query.max_results,
            min_relevance_score=query.min_relevance_score
        )
        
        # Use vector search with regulatory-focused query for now
        return await self.vector_search.search(regulatory_query)
    
    async def _execute_entity_search(self, query: RetrievalQuery):
        """Execute entity-focused definition search."""
        
        # Extract entities from query and search for their definitions
        entities = self._extract_entities_from_query(query.query_text)
        
        if not entities:
            return []
        
        # Create entity-focused query
        entity_query_text = f"definition of {' '.join(entities)} regulatory meaning"
        entity_query = RetrievalQuery(
            query_text=entity_query_text,
            query_type="entity",
            target_domains=query.target_domains,
            required_document_types=query.required_document_types,
            max_results=query.max_results,
            min_relevance_score=query.min_relevance_score
        )
        
        # Use keyword search for precise entity matching
        return await self.keyword_search.search(entity_query)
    
    def _extract_entities_from_query(self, query_text: str) -> List[str]:
        """Extract business/legal entities from query for entity search."""
        
        import re
        
        # Common entity patterns in compliance queries
        entity_patterns = [
            r'\b(?:collective\s+)?investment\s+(?:fund|scheme)\b',
            r'\b(?:special\s+purpose\s+)?vehicle\b',
            r'\bspv\b',
            r'\bfund\b',
            r'\bsyndicate\b',
            r'\b(?:investment\s+)?company\b',
            r'\bpartnership\b',
            r'\btrust\b',
            r'\bscheme\b'
        ]
        
        entities = []
        query_lower = query_text.lower()
        
        for pattern in entity_patterns:
            matches = re.finditer(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                entity = match.group(0).strip()
                if entity not in entities:
                    entities.append(entity)
        
        return entities
    
    def _get_relevant_document_types(self, context) -> List[DocumentType]:
        """Determine relevant document types based on context."""
        
        # Always include primary regulatory documents
        doc_types = [DocumentType.LAW, DocumentType.REGULATION, DocumentType.RULEBOOK]
        
        # Add guidance for practical implementation questions
        if any(domain in context.regulatory_domains for domain in 
               ['conduct_of_business', 'aml_sanctions']):
            doc_types.extend([DocumentType.GUIDANCE, DocumentType.CIRCULAR])
        
        return doc_types
    
    def _consolidate_results(self, all_results: List[HybridRetrievalResult]) -> Dict[str, Any]:
        """Consolidate results from multiple queries."""
        
        if not all_results:
            return {"documents": [], "total_queries": 0, "avg_confidence": 0.0}
        
        # Collect all unique documents
        all_documents = []
        seen_doc_ids = set()
        
        for result in all_results:
            for doc in result.documents:
                if doc.source.document_id not in seen_doc_ids:
                    all_documents.append(doc)
                    seen_doc_ids.add(doc.source.document_id)
        
        # Sort by relevance score
        all_documents.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Calculate average confidence
        avg_confidence = sum(r.fusion_confidence for r in all_results) / len(all_results)
        
        return {
            "documents": all_documents[:20],  # Limit to top 20 overall
            "total_queries": len(all_results),
            "avg_confidence": avg_confidence,
            "total_documents_found": len(all_documents),
            "query_coverage": sum(r.coverage_assessment for r in all_results) / len(all_results)
        }
    
    def _generate_summary(self, all_results: List[HybridRetrievalResult]) -> Dict[str, Any]:
        """Generate summary statistics for retrieval process."""
        
        if not all_results:
            return {"error": "No retrieval results"}
        
        total_docs_retrieved = sum(len(r.documents) for r in all_results)
        successful_queries = sum(1 for r in all_results if len(r.documents) > 0)
        
        # Method performance
        method_performance = {}
        for result in all_results:
            for method, stats in result.retrieval_stats.get('method_contributions', {}).items():
                if method not in method_performance:
                    method_performance[method] = {'total_retrieved': 0, 'in_final': 0}
                method_performance[method]['total_retrieved'] += stats.get('retrieved', 0)
                method_performance[method]['in_final'] += stats.get('in_final', 0)
        
        return {
            "queries_processed": len(all_results),
            "successful_queries": successful_queries,
            "total_documents_retrieved": total_docs_retrieved,
            "average_documents_per_query": total_docs_retrieved / len(all_results) if all_results else 0,
            "average_fusion_confidence": sum(r.fusion_confidence for r in all_results) / len(all_results),
            "method_performance": method_performance
        }
