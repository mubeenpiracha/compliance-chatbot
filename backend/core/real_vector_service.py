"""
Real Pinecone vector service that connects to your indexed documents.
"""
import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from pathlib import Path
from .config import PINECONE_API_KEY

logger = logging.getLogger(__name__)

class RealVectorService:
    """Real vector service that connects to Pinecone index with your ingested documents."""
    
    def __init__(self, index_name: str = "compliance-bot-index"):
        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY not set")
            
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(index_name)
        self.content_store = Path("./content_store")
        
        logger.info(f"Connected to Pinecone index: {index_name}")
    
    async def search(self, query_vector: List[float], top_k: int = 10, 
                    filter_params: Dict = None) -> List[Dict[str, Any]]:
        """Search the Pinecone index for similar documents across relevant namespaces."""
        try:
            all_results = []
            
            # Search across all namespaces instead of hardcoded ones
            # This avoids namespace collision issues after our ingestion fix
            try:
                response = self.index.query(
                    vector=query_vector,
                    top_k=top_k,
                    include_metadata=True
                    # Note: No namespace parameter = search all namespaces
                )
                
                for match in response.matches:
                    # Extract namespace from the match if available
                    namespace = getattr(match, 'namespace', 'unknown')
                    result = self._convert_match_to_result(match, namespace)
                    if result:
                        all_results.append(result)
                        
            except Exception as e:
                logger.error(f"Failed to search Pinecone index: {str(e)}")
                return []
            
            # Sort by relevance score and limit results
            all_results.sort(key=lambda x: x['score'], reverse=True)
            final_results = all_results[:top_k]
            
            logger.info(f"Vector search returned {len(final_results)} results from {len(set(r.get('namespace') for r in final_results))} namespaces")
            return final_results
            
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            return []
    
    def _convert_match_to_result(self, match, namespace: str) -> Optional[Dict[str, Any]]:
        """Convert Pinecone match to expected result format."""
        try:
            # Load the actual content from content store
            content = self._load_content_from_store(match.metadata)
            
            if not content or content == "Content not available":
                return None
            
            result = {
                'id': match.id,
                'content': content,
                'score': float(match.score),
                'namespace': namespace,
                'metadata': {
                    'title': self._extract_title(match.metadata),
                    'document_type': self._extract_document_type(match.metadata),
                    'section': match.metadata.get('section_path', ''),
                    'authority_level': self._determine_authority_level(match.metadata),
                    'jurisdiction': match.metadata.get('jurisdiction', 'ADGM'),
                    'page': match.metadata.get('page'),
                    'file_name': match.metadata.get('file_name'),
                    'source_path': match.metadata.get('source_path'),
                    'chunk_index': match.metadata.get('chunk_index'),
                    'checksum': match.metadata.get('checksum')
                }
            }
            return result
            
        except Exception as e:
            logger.error(f"Failed to convert match: {str(e)}")
            return None
    
    def _load_content_from_store(self, metadata: Dict) -> str:
        """Load the actual document content from content store."""
        try:
            content_uri = metadata.get('content_uri')
            if content_uri and Path(content_uri).exists():
                return Path(content_uri).read_text(encoding='utf-8')
            else:
                # Fallback: try to construct path from metadata
                file_name = metadata.get('file_name', '')
                chunk_index = metadata.get('chunk_index', 0)
                checksum = metadata.get('checksum', '')
                source_path = metadata.get('source_path', '')
                
                # Generate namespace like ingest.py does
                from ..scripts.ingest import generate_namespace_for_file
                namespace = generate_namespace_for_file(file_name, source_path)
                
                blob_path = self.content_store / namespace / f"{chunk_index:06d}_{checksum}.txt"
                if blob_path.exists():
                    return blob_path.read_text(encoding='utf-8')
                else:
                    logger.warning(f"Content file not found: {blob_path}")
                    return "Content not available"
                    
        except Exception as e:
            logger.error(f"Failed to load content: {str(e)}")
            return "Error loading content"
    
    def _extract_title(self, metadata: Dict) -> str:
        """Extract a readable title from the file name."""
        file_name = metadata.get('file_name', 'Unknown Document')
        # Clean up the file name to make it more readable
        title = file_name.replace('.pdf', '').replace('-', ' ').replace('_', ' ')
        return ' '.join(word.capitalize() for word in title.split())
    
    def _extract_document_type(self, metadata: Dict) -> str:
        """Determine document type from file name or path."""
        file_name = metadata.get('file_name', '').lower()
        
        if 'rulebook' in file_name:
            return 'rulebook'
        elif 'regulation' in file_name:
            return 'regulation'
        elif 'guidance' in file_name:
            return 'guidance'
        elif 'law' in file_name:
            return 'law'
        elif 'rule' in file_name:
            return 'rules'
        else:
            return 'document'
    
    def _determine_authority_level(self, metadata: Dict) -> int:
        """Determine authority level based on document type."""
        doc_type = self._extract_document_type(metadata)
        
        authority_map = {
            'law': 1,          # Highest authority
            'regulation': 2,    # High authority
            'rulebook': 3,     # Medium-high authority
            'rules': 3,        # Medium-high authority
            'guidance': 4,     # Lower authority
            'document': 5      # Lowest authority
        }
        
        return authority_map.get(doc_type, 5)
    
    def get_embedding(self, text: str) -> List[float]:
        """This method is not used in the current implementation but kept for compatibility."""
        # The actual embeddings are generated during ingestion
        # This would require OpenAI API call which is handled by the retrieval engines
        raise NotImplementedError("Use retrieval engines for embedding generation")


# For backward compatibility, create an alias
VectorService = RealVectorService
