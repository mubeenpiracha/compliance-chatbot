"""
BM25/Keyword search implementation for exact term matching.
"""
import re
from typing import List, Dict, Any, Optional
from collections import Counter
import math
from pathlib import Path
from ..models.retrieval_models import RetrievalQuery, RetrievedDocument, DocumentSource
import logging

logger = logging.getLogger(__name__)


class KeywordSearchEngine:
    """Handles keyword-based search using BM25 algorithm."""
    
    def __init__(self, document_corpus: List[Dict], vector_service=None):
        self.corpus = document_corpus
        self.vector_service = vector_service  # Optional Pinecone service for metadata lookup
        self.legal_synonyms = self._build_legal_synonyms()
        self.regulatory_terms = self._build_regulatory_terms()
        self._metadata_cache = {}  # Cache for Pinecone metadata lookups
        
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
            
            # Check jurisdiction (case-insensitive)
            if query.target_domains:
                doc_jurisdiction = metadata.get('jurisdiction', '').lower()
                target_domains_lower = [domain.lower() for domain in query.target_domains]
                if doc_jurisdiction not in target_domains_lower:
                    continue
            
            filtered_results.append((doc_data, score))
            
        return filtered_results
    
    async def _convert_to_retrieved_document(self, doc_data: Dict, score: float,
                                           query: RetrievalQuery, 
                                           query_terms: List[str]) -> RetrievedDocument:
        """Convert search result to RetrievedDocument with Pinecone metadata when available."""
        
        metadata = doc_data.get('metadata', {})
        
        # Try to get enhanced metadata from Pinecone if available
        content_path = doc_data.get('content_path') or metadata.get('content_path')
        pinecone_metadata = None
        
        if content_path and self.vector_service:
            pinecone_metadata = await self._get_pinecone_metadata_for_chunk(content_path)
        
        # Use Pinecone metadata as primary source, fall back to local metadata
        if pinecone_metadata:
            # Use Pinecone metadata for consistency with vector search
            enhanced_metadata = {
                'title': self._extract_title_from_pinecone(pinecone_metadata),
                'section': pinecone_metadata.get('section_path', ''),
                'subsection': pinecone_metadata.get('subsection_path'),
                'authority_level': self._determine_authority_level_from_pinecone(pinecone_metadata),
                'jurisdiction': pinecone_metadata.get('jurisdiction', 'ADGM'),
                'document_type': self._extract_document_type_from_pinecone(pinecone_metadata),
                'page': pinecone_metadata.get('page'),
                'file_name': pinecone_metadata.get('file_name'),
                'source_path': pinecone_metadata.get('source_path'),
                'chunk_index': pinecone_metadata.get('chunk_index'),
                'checksum': pinecone_metadata.get('checksum'),
                'content_uri': pinecone_metadata.get('content_uri')
            }
            logger.debug(f"Using Pinecone metadata for document {doc_data['id']}")
        else:
            # Fall back to local metadata (existing behavior)
            enhanced_metadata = {
                'title': metadata.get('title', 'Unknown Document'),
                'section': metadata.get('section'),
                'subsection': metadata.get('subsection'),
                'authority_level': metadata.get('authority_level', 4),
                'jurisdiction': metadata.get('jurisdiction', 'adgm'),
                'document_type': metadata.get('document_type', 'guidance'),
                'checksum': metadata.get('checksum') or metadata.get('chunk_id')
            }
            logger.debug(f"Using local metadata for document {doc_data['id']}")
        
        # Create document source with enhanced metadata
        source = DocumentSource(
            document_id=doc_data['id'],
            title=enhanced_metadata['title'],
            section=enhanced_metadata['section'],
            subsection=enhanced_metadata.get('subsection'),
            authority_level=enhanced_metadata.get('authority_level', 4),
            jurisdiction=enhanced_metadata.get('jurisdiction', 'ADGM'),
            chunk_id=enhanced_metadata.get('checksum') or enhanced_metadata.get('chunk_id')
        )
        
        # Generate keyword highlights
        highlights = self._generate_keyword_highlights(
            doc_data['content'], query_terms
        )
        
        return RetrievedDocument(
            source=source,
            content=doc_data['content'],
            relevance_score=min(score / 5.0, 1.0),  # Better normalize score for BM25 range
            retrieval_method="keyword",
            match_highlights=highlights,
            context_window=doc_data.get('context', doc_data['content'])
        )
    
    def _extract_title_from_pinecone(self, metadata: Dict) -> str:
        """Extract a readable title from Pinecone metadata."""
        file_name = metadata.get('file_name', 'Unknown Document')
        # Clean up the file name to make it more readable
        title = file_name.replace('.pdf', '').replace('-', ' ').replace('_', ' ')
        return ' '.join(word.capitalize() for word in title.split())
    
    def _extract_document_type_from_pinecone(self, metadata: Dict) -> str:
        """Determine document type from Pinecone metadata."""
        file_name = metadata.get('file_name', '').lower()
        
        if 'rulebook' in file_name:
            return 'rulebook'
        elif 'regulation' in file_name:
            return 'regulation'
        elif 'guidance' in file_name:
            return 'guidance'
        elif 'law' in file_name:
            return 'law'
        elif 'circular' in file_name:
            return 'circular'
        elif 'policy' in file_name:
            return 'policy'
        else:
            return 'guidance'
    
    def _determine_authority_level_from_pinecone(self, metadata: Dict) -> int:
        """Determine authority level from Pinecone metadata."""
        document_type = self._extract_document_type_from_pinecone(metadata)
        
        authority_levels = {
            'law': 1,
            'regulation': 2, 
            'rulebook': 3,
            'guidance': 4,
            'circular': 4,
            'policy': 5
        }
        
        return authority_levels.get(document_type, 4)
    
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

    async def _get_pinecone_metadata_for_chunk(self, content_path: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve Pinecone metadata for a chunk based on its content path.
        This ensures metadata consistency between vector and keyword search.
        """
        if not self.vector_service:
            return None
            
        # Check cache first
        if content_path in self._metadata_cache:
            return self._metadata_cache[content_path]
            
        try:
            # Extract chunk info from path like: /home/mubeen/compliance-chatbot/content_store/conduct-of-business-module-cob-difc/000595_482c28146be8ca607e10c321adf194a0fca6803f.txt
            path_obj = Path(content_path)
            if not path_obj.exists():
                logger.warning(f"Content path does not exist: {content_path}")
                return None
                
            # Parse filename to get chunk index and checksum
            filename_parts = path_obj.stem.split('_')
            if len(filename_parts) < 2:
                logger.warning(f"Invalid filename format: {path_obj.name}")
                return None
                
            chunk_index = filename_parts[0]
            checksum = filename_parts[1]
            namespace = path_obj.parent.name  # e.g., "conduct-of-business-module-cob-difc"
            
            # Search Pinecone for this specific chunk
            # We'll do a metadata-only query to find the chunk by its identifiers
            try:
                # Query Pinecone for documents with matching metadata
                # This is a bit of a workaround since Pinecone doesn't have direct metadata-only search
                # We'll search in the specific namespace with a dummy vector (all zeros)
                dummy_vector = [0.0] * 1536  # Standard OpenAI embedding size
                
                response = self.vector_service.index.query(
                    vector=dummy_vector,
                    top_k=1000,  # Get many results to find our specific chunk
                    include_metadata=True,
                    namespace=namespace,
                    filter={
                        "chunk_index": int(chunk_index),
                        "checksum": checksum
                    }
                )
                
                # Find the matching chunk
                for match in response.matches:
                    if (match.metadata.get('chunk_index') == int(chunk_index) and 
                        match.metadata.get('checksum') == checksum):
                        
                        # Cache the result
                        metadata = match.metadata
                        self._metadata_cache[content_path] = metadata
                        logger.debug(f"Found Pinecone metadata for {content_path}")
                        return metadata
                        
            except Exception as e:
                logger.debug(f"Pinecone metadata lookup failed for {content_path}: {str(e)}")
                
            # If direct filter search fails, try a different approach
            # Search by content_uri if it exists in the metadata
            try:
                response = self.vector_service.index.query(
                    vector=dummy_vector,
                    top_k=1000,
                    include_metadata=True,
                    namespace=namespace,
                    filter={"content_uri": content_path}
                )
                
                if response.matches:
                    metadata = response.matches[0].metadata
                    self._metadata_cache[content_path] = metadata
                    logger.debug(f"Found Pinecone metadata by content_uri for {content_path}")
                    return metadata
                    
            except Exception as e:
                logger.debug(f"Pinecone content_uri lookup failed for {content_path}: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error getting Pinecone metadata for {content_path}: {str(e)}")
            
        return None
