"""
Result fusion system for combining multiple retrieval strategies.
"""
import math
from typing import List, Dict, Any, Set
from collections import defaultdict
from ..models.retrieval_models import RetrievedDocument, HybridRetrievalResult, RetrievalQuery
import logging

logger = logging.getLogger(__name__)


class ResultFusion:
    """Combines and ranks results from multiple retrieval strategies."""
    
    def __init__(self):
        self.authority_weights = {
            1: 1.5,  # Laws
            2: 1.3,  # Regulations  
            3: 1.1,  # Rulebooks
            4: 1.0   # Guidance
        }
        
    def fuse_results(self, retrieval_results: Dict[str, List[RetrievedDocument]], 
                    query: RetrievalQuery) -> HybridRetrievalResult:
        """Fuse results from multiple retrieval methods."""
        
        try:
            # Apply Reciprocal Rank Fusion (RRF)
            rrf_scores = self._apply_reciprocal_rank_fusion(retrieval_results)
            
            # Apply authority weighting
            authority_weighted_scores = self._apply_authority_weighting(rrf_scores)
            
            # Apply recency weighting
            recency_weighted_scores = self._apply_recency_weighting(authority_weighted_scores)
            
            # Remove duplicates and near-duplicates
            deduplicated_results = self._remove_duplicates(recency_weighted_scores)
            
            # Final ranking and selection
            final_documents = self._final_ranking(deduplicated_results, query.max_results)
            
            # Calculate fusion confidence
            fusion_confidence = self._calculate_fusion_confidence(retrieval_results, final_documents)
            
            # Calculate coverage assessment
            coverage_assessment = self._assess_coverage(retrieval_results, final_documents)
            
            # Generate retrieval statistics
            stats = self._generate_statistics(retrieval_results, final_documents)
            
            return HybridRetrievalResult(
                query=query,
                documents=final_documents,
                retrieval_stats=stats,
                coverage_assessment=coverage_assessment,
                fusion_confidence=fusion_confidence
            )
            
        except Exception as e:
            logger.error(f"Result fusion failed: {str(e)}")
            # Return empty result with error info
            return HybridRetrievalResult(
                query=query,
                documents=[],
                retrieval_stats={"error": str(e)},
                coverage_assessment=0.0,
                fusion_confidence=0.0
            )
    
    def _apply_reciprocal_rank_fusion(self, 
                                    retrieval_results: Dict[str, List[RetrievedDocument]]) -> Dict[str, Dict[str, float]]:
        """Apply Reciprocal Rank Fusion to combine rankings."""
        
        rrf_scores = defaultdict(lambda: defaultdict(float))
        k = 60  # RRF constant
        
        for method, documents in retrieval_results.items():
            for rank, doc in enumerate(documents, 1):
                doc_id = doc.source.document_id
                # RRF formula: 1 / (k + rank)
                rrf_score = 1.0 / (k + rank)
                rrf_scores[doc_id][method] = rrf_score
                rrf_scores[doc_id]['document'] = doc  # Store document reference
        
        return rrf_scores
    
    def _apply_authority_weighting(self, 
                                 rrf_scores: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Apply document authority weighting."""
        
        weighted_scores = {}
        
        for doc_id, scores in rrf_scores.items():
            document = scores['document']
            authority_weight = self.authority_weights.get(document.source.authority_level, 1.0)
            
            weighted_scores[doc_id] = {
                'total_score': sum(score for key, score in scores.items() if key != 'document') * authority_weight,
                'authority_weight': authority_weight,
                'document': document,
                'method_scores': {k: v for k, v in scores.items() if k != 'document'}
            }
        
        return weighted_scores
    
    def _apply_recency_weighting(self, 
                               weighted_scores: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Apply recency weighting (newer documents get slight boost)."""
        
        for doc_id, score_data in weighted_scores.items():
            document = score_data['document']
            
            # Simple recency boost - could be enhanced with actual date analysis
            recency_boost = 1.0
            if hasattr(document.source, 'last_updated') and document.source.last_updated:
                # Boost newer documents slightly (this would need actual date comparison)
                recency_boost = 1.05
            
            score_data['total_score'] *= recency_boost
            score_data['recency_boost'] = recency_boost
        
        return weighted_scores
    
    def _remove_duplicates(self, 
                         weighted_scores: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Remove duplicate and near-duplicate documents."""
        
        # Group documents by similarity
        document_groups = self._group_similar_documents(weighted_scores)
        
        # Keep the highest-scoring document from each group
        deduplicated = {}
        for group in document_groups:
            if group:  # Skip empty groups
                best_doc_id = max(group, key=lambda doc_id: weighted_scores[doc_id]['total_score'])
                deduplicated[best_doc_id] = weighted_scores[best_doc_id]
        
        return deduplicated
    
    def _group_similar_documents(self, 
                               weighted_scores: Dict[str, Dict[str, float]]) -> List[List[str]]:
        """Group similar documents together."""
        
        doc_ids = list(weighted_scores.keys())
        groups = []
        processed = set()
        
        for doc_id in doc_ids:
            if doc_id in processed:
                continue
                
            current_group = [doc_id]
            processed.add(doc_id)
            
            doc1 = weighted_scores[doc_id]['document']
            
            for other_id in doc_ids:
                if other_id in processed:
                    continue
                    
                doc2 = weighted_scores[other_id]['document']
                
                if self._are_similar_documents(doc1, doc2):
                    current_group.append(other_id)
                    processed.add(other_id)
            
            groups.append(current_group)
        
        return groups
    
    def _are_similar_documents(self, doc1: RetrievedDocument, 
                             doc2: RetrievedDocument) -> bool:
        """Check if two documents are similar enough to be considered duplicates."""
        
        # Same document from same source
        if (doc1.source.document_id == doc2.source.document_id or
            (doc1.source.title == doc2.source.title and 
             doc1.source.section == doc2.source.section)):
            return True
        
        # Very similar content (simple heuristic)
        if len(doc1.content) > 100 and len(doc2.content) > 100:
            # Check if significant portion of content overlaps
            shorter_content = doc1.content if len(doc1.content) < len(doc2.content) else doc2.content
            longer_content = doc2.content if len(doc1.content) < len(doc2.content) else doc1.content
            
            if shorter_content[:200] in longer_content:
                return True
        
        return False
    
    def _final_ranking(self, deduplicated_results: Dict[str, Dict[str, float]], 
                      max_results: int) -> List[RetrievedDocument]:
        """Apply final ranking and select top results."""
        
        # Sort by total score
        sorted_docs = sorted(
            deduplicated_results.items(),
            key=lambda x: x[1]['total_score'],
            reverse=True
        )
        
        # Select top documents and update their fusion scores
        final_documents = []
        for doc_id, score_data in sorted_docs[:max_results]:
            document = score_data['document']
            # Update relevance score with fusion score
            document.relevance_score = min(score_data['total_score'], 1.0)
            document.retrieval_method = "fusion"
            final_documents.append(document)
        
        return final_documents
    
    def _calculate_fusion_confidence(self, 
                                   retrieval_results: Dict[str, List[RetrievedDocument]],
                                   final_documents: List[RetrievedDocument]) -> float:
        """Calculate confidence in the fusion process."""
        
        if not final_documents:
            return 0.0
        
        # Base confidence on method agreement
        method_count = len(retrieval_results)
        if method_count == 0:
            return 0.0
        
        # Check how many methods contributed to final results
        contributing_methods = set()
        for doc in final_documents:
            for method, method_docs in retrieval_results.items():
                if any(d.source.document_id == doc.source.document_id for d in method_docs):
                    contributing_methods.add(method)
        
        method_coverage = len(contributing_methods) / method_count
        
        # Factor in average relevance score of final results
        avg_relevance = sum(doc.relevance_score for doc in final_documents) / len(final_documents)
        
        # Combine factors
        confidence = (method_coverage * 0.6) + (avg_relevance * 0.4)
        return min(confidence, 1.0)
    
    def _assess_coverage(self, 
                        retrieval_results: Dict[str, List[RetrievedDocument]],
                        final_documents: List[RetrievedDocument]) -> float:
        """Assess how well the final results cover the query."""
        
        if not retrieval_results:
            return 0.0
        
        # Count unique documents across all methods
        all_unique_docs = set()
        for method_docs in retrieval_results.values():
            for doc in method_docs:
                all_unique_docs.add(doc.source.document_id)
        
        # Check coverage in final results
        final_doc_ids = set(doc.source.document_id for doc in final_documents)
        
        if not all_unique_docs:
            return 0.0
        
        coverage = len(final_doc_ids.intersection(all_unique_docs)) / len(all_unique_docs)
        return min(coverage, 1.0)
    
    def _generate_statistics(self, 
                           retrieval_results: Dict[str, List[RetrievedDocument]],
                           final_documents: List[RetrievedDocument]) -> Dict[str, Any]:
        """Generate retrieval statistics."""
        
        stats = {
            'methods_used': list(retrieval_results.keys()),
            'total_retrieved': sum(len(docs) for docs in retrieval_results.values()),
            'final_count': len(final_documents),
            'method_contributions': {}
        }
        
        # Count contributions by method
        for method, method_docs in retrieval_results.items():
            method_doc_ids = set(doc.source.document_id for doc in method_docs)
            final_doc_ids = set(doc.source.document_id for doc in final_documents)
            
            stats['method_contributions'][method] = {
                'retrieved': len(method_docs),
                'in_final': len(method_doc_ids.intersection(final_doc_ids)),
                'avg_score': sum(doc.relevance_score for doc in method_docs) / len(method_docs) if method_docs else 0
            }
        
        return stats
