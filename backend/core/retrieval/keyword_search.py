"""
BM25/Keyword search implementation for exact term matching.
"""
import re
from typing import List, Dict, Any
from collections import Counter
import math
from ..models.retrieval_models import RetrievalQuery, RetrievedDocument, DocumentSource
import logging

logger = logging.getLogger(__name__)


class KeywordSearchEngine:
    """Handles keyword-based search using BM25 algorithm."""
    
    def __init__(self, document_corpus: List[Dict]):
        self.corpus = document_corpus
        self.legal_synonyms = self._build_legal_synonyms()
        self.regulatory_terms = self._build_regulatory_terms()
        
    async def search(self, query: RetrievalQuery) -> List[RetrievedDocument]:
        """Perform BM25 keyword search."""
        
        try:
            # Expand query with legal synonyms and regulatory terms
            expanded_query = self._expand_query(query.query_text)
            
            # Tokenize and preprocess query
            query_terms = self._preprocess_text(expanded_query)
            
            # Calculate BM25 scores for all documents
            scored_documents = []
            for doc in self.corpus:
                score = self._calculate_bm25_score(query_terms, doc)
                if score >= query.min_relevance_score:
                    scored_documents.append((doc, score))
            
            # Sort by relevance and apply filters
            scored_documents.sort(key=lambda x: x[1], reverse=True)
            filtered_docs = self._apply_filters(scored_documents, query)
            
            # Convert to RetrievedDocument objects
            documents = []
            for doc_data, score in filtered_docs[:query.max_results]:
                retrieved_doc = await self._convert_to_retrieved_document(
                    doc_data, score, query, query_terms
                )
                documents.append(retrieved_doc)
            
            logger.info(f"Keyword search returned {len(documents)} documents for query: {query.query_text[:50]}...")
            return documents
            
        except Exception as e:
            logger.error(f"Keyword search failed: {str(e)}")
            return []
    
    def _expand_query(self, query_text: str) -> str:
        """Expand query with legal synonyms and related terms."""
        
        expanded_terms = [query_text]
        query_lower = query_text.lower()
        
        # Add synonyms
        for term, synonyms in self.legal_synonyms.items():
            if term in query_lower:
                expanded_terms.extend(synonyms)
        
        # Add regulatory term variations
        for base_term, variations in self.regulatory_terms.items():
            if base_term in query_lower:
                expanded_terms.extend(variations)
        
        return ' '.join(expanded_terms)
    
    def _build_legal_synonyms(self) -> Dict[str, List[str]]:
        """Build dictionary of legal term synonyms."""
        
        return {
            'fund': ['collective investment scheme', 'cis', 'investment fund', 'pooled fund'],
            'spv': ['special purpose vehicle', 'special purpose entity', 'spe'],
            'license': ['licence', 'permit', 'authorization', 'approval'],
            'requirement': ['obligation', 'mandate', 'condition', 'provision'],
            'regulation': ['rule', 'law', 'statute', 'ordinance'],
            'syndicate': ['consortium', 'joint venture', 'partnership', 'alliance']
        }
    
    def _build_regulatory_terms(self) -> Dict[str, List[str]]:
        """Build dictionary of regulatory term variations."""
        
        return {
            'aml': ['anti-money laundering', 'anti money laundering'],
            'kyc': ['know your customer', 'know your client'],
            'cdd': ['customer due diligence', 'client due diligence'],
            'fsp': ['financial services permission'],
            'adgm': ['abu dhabi global market'],
            'difc': ['dubai international financial centre'],
            'fsra': ['financial services regulatory authority']
        }
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for keyword matching."""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation but keep hyphens in compound terms
        text = re.sub(r'[^\w\s\-]', ' ', text)
        
        # Split into terms
        terms = text.split()
        
        # Remove common stop words (but keep important regulatory terms)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        terms = [term for term in terms if term not in stop_words or len(term) > 3]
        
        return terms
    
    def _calculate_bm25_score(self, query_terms: List[str], document: Dict) -> float:
        """Calculate BM25 relevance score."""
        
        # BM25 parameters
        k1 = 1.5
        b = 0.75
        
        # Document content
        doc_content = document.get('content', '')
        doc_terms = self._preprocess_text(doc_content)
        doc_length = len(doc_terms)
        
        # Average document length (approximate)
        avg_doc_length = 500  # Could be calculated from corpus
        
        # Calculate term frequencies
        doc_term_freq = Counter(doc_terms)
        
        score = 0.0
        for term in query_terms:
            if term in doc_term_freq:
                # Term frequency
                tf = doc_term_freq[term]
                
                # Document frequency (simplified - using inverse frequency boost)
                df_boost = 1.0  # Could be calculated from corpus
                
                # BM25 formula
                term_score = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length)))
                term_score *= df_boost
                
                score += term_score
        
        # Boost for exact phrase matches
        if len(query_terms) > 1:
            query_phrase = ' '.join(query_terms)
            if query_phrase in doc_content.lower():
                score *= 1.5
        
        # Boost for document authority
        authority_boost = self._get_authority_boost(document)
        score *= authority_boost
        
        return score
    
    def _get_authority_boost(self, document: Dict) -> float:
        """Apply authority-based boost to relevance score."""
        
        doc_type = document.get('metadata', {}).get('document_type', 'guidance')
        
        authority_multipliers = {
            'law': 1.5,
            'regulation': 1.4,
            'rulebook': 1.3,
            'guidance': 1.0,
            'circular': 0.9,
            'policy': 0.8
        }
        
        return authority_multipliers.get(doc_type, 1.0)
    
    def _apply_filters(self, scored_docs: List, query: RetrievalQuery) -> List:
        """Apply jurisdiction and document type filters."""
        
        filtered_results = []
        
        for doc_data, score in scored_docs:
            metadata = doc_data.get('metadata', {})
            
            # Check jurisdiction
            if query.target_domains and metadata.get('jurisdiction') not in query.target_domains:
                continue
            
            filtered_results.append((doc_data, score))
            
        return filtered_results
    
    async def _convert_to_retrieved_document(self, doc_data: Dict, score: float,
                                           query: RetrievalQuery, 
                                           query_terms: List[str]) -> RetrievedDocument:
        """Convert search result to RetrievedDocument."""
        
        metadata = doc_data.get('metadata', {})
        
        # Create document source
        source = DocumentSource(
            document_id=doc_data['id'],

            title=metadata.get('title', 'Unknown Document'),
            section=metadata.get('section'),
            subsection=metadata.get('subsection'),
            authority_level=metadata.get('authority_level', 4),
            jurisdiction=metadata.get('jurisdiction', 'adgm'),
            chunk_id=metadata.get('checksum') or metadata.get('chunk_id')
        )
        
        # Generate keyword highlights
        highlights = self._generate_keyword_highlights(
            doc_data['content'], query_terms
        )
        
        return RetrievedDocument(
            source=source,
            content=doc_data['content'],
            relevance_score=min(score / 10.0, 1.0),  # Normalize score
            retrieval_method="keyword",
            match_highlights=highlights,
            context_window=doc_data.get('context', doc_data['content'])
        )
    
    def _generate_keyword_highlights(self, content: str, 
                                   query_terms: List[str]) -> List[str]:
        """Generate highlights showing keyword matches."""
        
        highlights = []
        content_lower = content.lower()
        
        # Find sentences with keyword matches
        sentences = re.split(r'[.!?]+', content)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            matches = sum(1 for term in query_terms if term in sentence_lower)
            
            if matches > 0:
                # Highlight matching terms
                highlighted = sentence
                for term in query_terms:
                    pattern = re.compile(re.escape(term), re.IGNORECASE)
                    highlighted = pattern.sub(f'**{term}**', highlighted)
                
                if len(highlighted) > 300:
                    highlighted = highlighted[:300] + "..."
                
                highlights.append(highlighted.strip())
                
        return highlights[:5]  # Return top 5 highlights
