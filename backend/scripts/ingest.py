# backend/scripts/ingest.py
import os
import sys
import time
import re
import json
import hashlib
import numpy as np
import tiktoken
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Document
from llama_index.readers.file import UnstructuredReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.node_parser import (
    SentenceSplitter, 
    SemanticSplitterNodeParser,
    MarkdownNodeParser,
    HierarchicalNodeParser
)
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import BaseNode, TextNode
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import PINECONE_API_KEY, OPENAI_API_KEY

# Enhanced imports for state-of-the-art techniques
try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
    ENHANCED_MODE = True
    print("Enhanced chunking mode: ENABLED")
except ImportError:
    ENHANCED_MODE = False
    print("Enhanced chunking mode: DISABLED (install sentence-transformers and scikit-learn)")

class ChunkingStrategy(Enum):
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical" 
    HYBRID = "hybrid"
    LATE_CHUNKING = "late_chunking"
    TOKEN_AWARE = "token_aware"

class DocumentType(Enum):
    REGULATORY_PROSE = "regulatory_prose"  # Rules, regulations
    NARRATIVE_GUIDANCE = "narrative_guidance"  # Explanatory documents
    BULLET_DEFINITIONS = "bullet_definitions"  # Lists, definitions
    TABLES = "tables"  # Structured data
    OTHER = "other"

@dataclass
class TokenAwareChunkConfig:
    target_tokens: int
    overlap_percentage: float
    max_tokens: int
    boundary_window: int = 40  # Tokens to allow for boundary adjustment

@dataclass
class ChunkMetrics:
    coherence_score: float
    completeness_score: float
    boundary_quality: float
    information_density: float

# Set OpenAI API key for LlamaIndex components
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

class RegulatoryDocumentChunker:
    """
    Enhanced chunker for regulatory documents with state-of-the-art techniques.
    """
    
    def __init__(self, embedding_model, chunk_size: int = 1024, chunk_overlap: int = 128, enable_late_chunking: bool = True, enable_token_aware: bool = True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.enable_late_chunking = enable_late_chunking and ENHANCED_MODE
        self.enable_token_aware = enable_token_aware
        
        # Initialize tokenizer for token-aware chunking
        self.tokenizer = None
        if enable_token_aware:
            try:
                self.tokenizer = tiktoken.encoding_for_model("text-embedding-3-large")
                print("  ✓ Loaded tiktoken tokenizer for token-aware chunking")
            except:
                print("  Warning: Could not load tiktoken, using character-based chunking")
                self.enable_token_aware = False
        
        # Document-specific chunk configurations
        self.chunk_configs = {
            DocumentType.REGULATORY_PROSE: TokenAwareChunkConfig(
                target_tokens=525,  # Average of 450-600
                overlap_percentage=15.0,
                max_tokens=650,
                boundary_window=40
            ),
            DocumentType.NARRATIVE_GUIDANCE: TokenAwareChunkConfig(
                target_tokens=800,  # Average of 700-900
                overlap_percentage=11.0,
                max_tokens=950,
                boundary_window=40
            ),
            DocumentType.BULLET_DEFINITIONS: TokenAwareChunkConfig(
                target_tokens=475,  # Average of 400-550
                overlap_percentage=9.0,
                max_tokens=600,
                boundary_window=40
            ),
            DocumentType.TABLES: TokenAwareChunkConfig(
                target_tokens=600,  # Flexible for tables
                overlap_percentage=5.0,  # Minimal overlap for tables
                max_tokens=800,
                boundary_window=20
            ),
            DocumentType.OTHER: TokenAwareChunkConfig(
                target_tokens=600,  # Default
                overlap_percentage=12.0,
                max_tokens=750,
                boundary_window=40
            )
        }
        
        # Initialize advanced models if available
        self.sentence_transformer = None
        if ENHANCED_MODE:
            try:
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                print("  ✓ Loaded sentence transformer for enhanced chunking")
            except:
                print("  Warning: Could not load sentence transformer, using basic mode")
        
        # Initialize multiple chunking strategies
        self.semantic_splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=embedding_model
        )
        
        self.hierarchical_splitter = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 1024, 512],  # Increased minimum chunk size
            chunk_overlap=chunk_overlap
        )
        
        self.sentence_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=" ",
            paragraph_separator="\n\n",
            chunking_tokenizer_fn=self._custom_tokenizer
        )
    
    def _custom_tokenizer(self, text: str) -> List[str]:
        """Custom tokenizer that's more aware of regulatory document structure."""
        # Split on common regulatory patterns
        patterns = [
            r'\n(?=\d+\.\s)',  # Numbered sections
            r'\n(?=[A-Z][a-z]+:)',  # Section headers
            r'\n(?=\([a-z]\))',  # Subsections like (a), (b)
            r'\n(?=\([0-9]+\))',  # Numbered subsections like (1), (2)
            r'(?<=\.)\s+(?=[A-Z])',  # Sentence boundaries
        ]
        
        tokens = [text]
        for pattern in patterns:
            new_tokens = []
            for token in tokens:
                split_tokens = re.split(pattern, token)
                new_tokens.extend([t.strip() for t in split_tokens if t.strip()])
            tokens = new_tokens
        
        return tokens
    
    def extract_document_structure(self, text: str) -> dict:
        """Extract structural information from regulatory documents."""
        structure = {
            'sections': [],
            'subsections': [],
            'tables': [],
            'definitions': [],
            'references': []
        }
        
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Detect main sections (numbered sections)
            if re.match(r'^\d+\.\s+[A-Z]', line):
                structure['sections'].append({
                    'line_number': i,
                    'title': line,
                    'level': 1
                })
            
            # Detect subsections
            elif re.match(r'^\d+\.\d+\s+[A-Z]', line):
                structure['subsections'].append({
                    'line_number': i,
                    'title': line,
                    'level': 2
                })
            
            # Detect definitions
            elif 'definition' in line.lower() or 'means' in line.lower():
                structure['definitions'].append({
                    'line_number': i,
                    'text': line
                })
            
            # Detect table references
            elif re.search(r'table\s+\d+', line.lower()):
                structure['tables'].append({
                    'line_number': i,
                    'reference': line
                })
            
            # Detect cross-references
            elif re.search(r'section\s+\d+|paragraph\s+\d+|rule\s+\d+', line.lower()):
                structure['references'].append({
                    'line_number': i,
                    'reference': line
                })
        
        return structure
    
    def enhance_metadata(self, document: Document) -> Document:
        """Enhanced metadata extraction with size limits for Pinecone."""
        text = document.text
        structure = self.extract_document_structure(text)
        
        # Add core structural metadata (keep essential info only)
        document.metadata.update({
            'num_sections': len(structure['sections']),
            'num_subsections': len(structure['subsections']),
            'document_length': len(text),
            'estimated_reading_time': min(len(text.split()) // 200, 999),  # Cap at 999 minutes
        })
        
        # Extract document type from filename or content  
        filename = document.metadata.get('file_name', '').lower()
        if 'regulation' in filename or 'regulation' in text[:1000].lower():
            document.metadata['document_type'] = 'regulation'
        elif 'rulebook' in filename or 'rules' in text[:1000].lower():
            document.metadata['document_type'] = 'rulebook'
        elif 'guidance' in filename or 'guide' in filename:
            document.metadata['document_type'] = 'guidance'
        else:
            document.metadata['document_type'] = 'other'
        
        # Extract key regulatory concepts (limit to top 5)
        regulatory_keywords = [
            'compliance', 'violation', 'penalty', 'requirement', 'obligation',
            'prohibited', 'permitted', 'license', 'authorization', 'approval',
            'reporting', 'disclosure', 'audit', 'inspection', 'enforcement'
        ]
        
        found_keywords = []
        text_lower = text.lower()
        for keyword in regulatory_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
                if len(found_keywords) >= 5:  # Limit to 5 keywords
                    break
        
        document.metadata['regulatory_concepts'] = found_keywords
        
        # Remove large metadata fields that might cause size issues
        metadata_to_remove = ['coordinates', 'languages', 'filetype']
        for key in metadata_to_remove:
            document.metadata.pop(key, None)
        
        return document
    
    def late_chunking_strategy(self, document: Document, context_size: int = 4096) -> List[BaseNode]:
        """
        Implement late chunking: embed large contexts, then split while preserving embeddings.
        """
        if not self.enable_late_chunking:
            # Fallback to sentence splitting
            return self.sentence_splitter.get_nodes_from_documents([document])
        
        print(f"Applying late chunking to: {document.metadata.get('file_name', 'Unknown')}")
        
        text = document.text
        contexts = []
        step_size = context_size - self.chunk_overlap
        
        # Split into overlapping large contexts
        for i in range(0, len(text), step_size):
            context = text[i:i + context_size]
            if len(context.strip()) > 100:
                contexts.append({
                    'text': context,
                    'start_pos': i,
                    'end_pos': i + len(context)
                })
        
        # Create smaller chunks but preserve context information
        nodes = []
        for ctx_idx, ctx in enumerate(contexts):
            small_chunks = self._split_text_with_quality_metrics(ctx['text'])
            
            for chunk_idx, chunk_text in enumerate(small_chunks):
                if len(chunk_text.strip()) < 50:
                    continue
                
                node = TextNode(text=chunk_text)
                node.metadata.update({
                    **document.metadata,
                    'chunk_id': f"{document.hash}_{ctx_idx}_{chunk_idx}",
                    'context_start': ctx['start_pos'],
                    'context_end': ctx['end_pos'],
                    'chunk_in_context': chunk_idx,
                    'chunking_strategy': 'late_chunking',
                    'context_size': len(ctx['text'])
                })
                nodes.append(node)
        
        print(f"Late chunking created {len(nodes)} nodes from {len(contexts)} contexts")
        return nodes
    
    def _split_text_with_quality_metrics(self, text: str) -> List[str]:
        """Split text with quality assessment for optimal boundaries."""
        strategies = [
            self._split_by_sentences,
            self._split_by_paragraphs,
            self._split_by_semantic_boundaries if ENHANCED_MODE else self._split_by_sentences
        ]
        
        all_splits = []
        for strategy in strategies:
            try:
                splits = strategy(text)
                quality_score = self._assess_split_quality(text, splits)
                all_splits.append((splits, quality_score))
            except Exception as e:
                continue
        
        if not all_splits:
            return self._simple_split(text)
        
        best_splits, best_score = max(all_splits, key=lambda x: x[1])
        return best_splits
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split by sentences with regulatory awareness."""
        sentences = re.split(r'(?<=\.)\s+(?=[A-Z])', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split by paragraphs."""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk + para) <= self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks
    
    def _split_by_semantic_boundaries(self, text: str) -> List[str]:
        """Split by semantic boundaries using sentence similarity."""
        if not ENHANCED_MODE or not self.sentence_transformer:
            return self._split_by_sentences(text)
        
        sentences = re.split(r'(?<=\.)\s+(?=[A-Z])', text)
        if len(sentences) <= 3:
            return [text]
        
        try:
            # Calculate sentence embeddings
            embeddings = self.sentence_transformer.encode(sentences)
            
            # Calculate similarities between adjacent sentences
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
                similarities.append(sim)
            
            # Find break points where similarity is low
            if similarities:
                threshold = np.percentile(similarities, 25)
                break_points = [0]
                
                for i, sim in enumerate(similarities):
                    if sim < threshold:
                        break_points.append(i + 1)
                
                break_points.append(len(sentences))
                
                # Create chunks based on break points
                chunks = []
                for i in range(len(break_points) - 1):
                    chunk_sentences = sentences[break_points[i]:break_points[i + 1]]
                    chunk_text = ' '.join(chunk_sentences)
                    if len(chunk_text.strip()) > 50:
                        chunks.append(chunk_text.strip())
                
                return chunks
        except Exception as e:
            print(f"Semantic boundary detection failed: {e}")
        
        return self._split_by_sentences(text)
    
    def _simple_split(self, text: str) -> List[str]:
        """Simple fallback splitting method."""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks
    
    def _assess_split_quality(self, original_text: str, splits: List[str]) -> float:
        """Assess the quality of a text splitting strategy."""
        if not splits:
            return 0.0
        
        scores = []
        
        # Completeness
        total_chars = sum(len(split) for split in splits)
        completeness = min(1.0, total_chars / len(original_text))
        scores.append(completeness)
        
        # Size consistency
        if ENHANCED_MODE:
            sizes = [len(split) for split in splits]
            target_size = self.chunk_size
            size_variance = np.var([abs(size - target_size) for size in sizes])
            size_consistency = 1.0 / (1.0 + size_variance / 1000)
            scores.append(size_consistency)
        else:
            scores.append(0.8)  # Default good score
        
        # Boundary quality
        boundary_quality = 0.0
        good_endings = ['.', '!', '?', ':', ';', '\n\n']
        for split in splits:
            if any(split.rstrip().endswith(ending) for ending in good_endings):
                boundary_quality += 1.0
        boundary_quality /= len(splits)
        scores.append(boundary_quality)
        
        return sum(scores) / len(scores)
    
    def hybrid_chunking_strategy(self, document: Document) -> List[BaseNode]:
        """
        Hybrid strategy combining multiple approaches and selecting the best chunks.
        """
        print(f"Applying hybrid chunking to: {document.metadata.get('file_name', 'Unknown')}")
        
        strategies = {}
        
        # Always include basic strategies
        try:
            nodes = self.semantic_splitter.get_nodes_from_documents([document])
            for node in nodes:
                node.metadata['chunking_strategy'] = 'semantic'
            strategies['semantic'] = nodes
        except Exception as e:
            print(f"  Semantic chunking failed: {e}")
        
        try:
            nodes = self.hierarchical_splitter.get_nodes_from_documents([document])
            for node in nodes:
                node.metadata['chunking_strategy'] = 'hierarchical'
            strategies['hierarchical'] = nodes
        except Exception as e:
            print(f"  Hierarchical chunking failed: {e}")
        
        # Add late chunking if enabled
        if self.enable_late_chunking:
            try:
                nodes = self.late_chunking_strategy(document)
                strategies['late_chunking'] = nodes
            except Exception as e:
                print(f"  Late chunking failed: {e}")
        
        if not strategies:
            # Fallback to sentence splitting
            nodes = self.sentence_splitter.get_nodes_from_documents([document])
            for node in nodes:
                node.metadata['chunking_strategy'] = 'sentence'
            return nodes
        
        # Select best strategy based on document characteristics
        final_nodes = self._select_best_strategy(strategies, document)
        
        print(f"Hybrid chunking selected {len(final_nodes)} chunks")
        return final_nodes
    
    def _select_best_strategy(self, strategies: Dict[str, List[BaseNode]], document: Document) -> List[BaseNode]:
        """Select the best chunking strategy based on document characteristics."""
        if len(strategies) == 1:
            return list(strategies.values())[0]
        
        doc_length = len(document.text)
        doc_type = document.metadata.get('document_type', 'other')
        
        # Strategy selection logic
        if doc_length > 50000:  # Large documents
            return strategies.get('hierarchical', strategies.get('late_chunking', list(strategies.values())[0]))
        elif doc_type in ['regulation', 'rulebook']:  # Structured documents
            return strategies.get('semantic', strategies.get('hierarchical', list(strategies.values())[0]))
        elif self.enable_late_chunking and 'late_chunking' in strategies:  # Enhanced mode
            return strategies['late_chunking']
        else:
            return strategies.get('semantic', list(strategies.values())[0])
    
    def chunk_document(self, document: Document, strategy: ChunkingStrategy = ChunkingStrategy.TOKEN_AWARE) -> List[BaseNode]:
        """
        Apply intelligent chunking strategy with enhanced techniques.
        """
        # Enhance metadata first
        document = self.enhance_metadata(document)
        
        if strategy == ChunkingStrategy.TOKEN_AWARE:
            return self.token_aware_chunking_strategy(document)
        elif strategy == ChunkingStrategy.HYBRID:
            return self.hybrid_chunking_strategy(document)
        elif strategy == ChunkingStrategy.LATE_CHUNKING and self.enable_late_chunking:
            return self.late_chunking_strategy(document)
        elif strategy == ChunkingStrategy.SEMANTIC:
            nodes = self.semantic_splitter.get_nodes_from_documents([document])
            for node in nodes:
                node.metadata['chunking_strategy'] = 'semantic'
            return nodes
        elif strategy == ChunkingStrategy.HIERARCHICAL:
            nodes = self.hierarchical_splitter.get_nodes_from_documents([document])
            for node in nodes:
                node.metadata['chunking_strategy'] = 'hierarchical'
            return nodes
        else:
            # Default to token-aware
            return self.token_aware_chunking_strategy(document)
    
    def normalize_text(self, text: str) -> str:
        """Normalize text before chunking to handle PDF artifacts and formatting issues."""
        # Remove soft hyphens and fix hyphenated line breaks
        text = re.sub(r'\u00AD', '', text)  # Remove soft hyphens
        text = re.sub(r'-\s*\n\s*', '', text)  # Fix hyphenated line breaks
        
        # Collapse weird whitespace but keep paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r'\n[ \t]+', '\n', text)  # Leading whitespace after newlines
        text = re.sub(r'[ \t]+\n', '\n', text)  # Trailing whitespace before newlines
        text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double newlines
        
        return text.strip()
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the embedding model's tokenizer."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback to character-based estimation (roughly 4 chars per token)
            return len(text) // 4
    
    def classify_document_type(self, document: Document) -> DocumentType:
        """Classify the document type based on content and filename."""
        filename = document.metadata.get('file_name', '').lower()
        text_sample = document.text[:2000].lower()  # Sample first 2000 chars
        
        # Check filename patterns
        if 'regulation' in filename or 'rules' in filename:
            return DocumentType.REGULATORY_PROSE
        elif 'guidance' in filename or 'guide' in filename:
            return DocumentType.NARRATIVE_GUIDANCE
        elif 'table' in filename or 'schedule' in filename:
            return DocumentType.TABLES
        
        # Check content patterns
        bullet_patterns = [r'\n\s*[•·\-]\s+', r'\n\s*\([a-z]\)', r'\n\s*\d+\.\s+']
        definition_patterns = ['definition', 'means', 'refers to', 'shall mean']
        table_patterns = ['table', 'schedule', 'appendix', '│', '┌', '└']
        
        bullet_count = sum(len(re.findall(pattern, text_sample)) for pattern in bullet_patterns)
        definition_count = sum(text_sample.count(word) for word in definition_patterns)
        table_count = sum(text_sample.count(word) for word in table_patterns)
        
        if bullet_count > 10 or definition_count > 5:
            return DocumentType.BULLET_DEFINITIONS
        elif table_count > 3:
            return DocumentType.TABLES
        elif 'guidance' in text_sample or 'explanation' in text_sample:
            return DocumentType.NARRATIVE_GUIDANCE
        elif any(word in text_sample for word in ['shall', 'must', 'requirement', 'obligation']):
            return DocumentType.REGULATORY_PROSE
        else:
            return DocumentType.OTHER
    
    def find_structural_boundaries(self, text: str) -> List[int]:
        """Find structural boundaries in text that should not be split."""
        boundaries = []
        lines = text.split('\n')
        char_pos = 0
        
        for i, line in enumerate(lines):
            line_start = char_pos
            char_pos += len(line) + 1  # +1 for newline
            
            # Mark boundaries we shouldn't split
            stripped = line.strip()
            if not stripped:
                continue
                
            # Headings (numbered sections, titles)
            if re.match(r'^\d+\.\s+[A-Z]', stripped) or re.match(r'^[A-Z][^a-z]*$', stripped):
                boundaries.append((line_start, char_pos, 'heading'))
            # Bullet points
            elif re.match(r'^\s*[•·\-]\s+', line) or re.match(r'^\s*\([a-z]\)', line):
                boundaries.append((line_start, char_pos, 'bullet'))
            # Table rows (simplified detection)
            elif '│' in line or line.count('|') >= 2:
                boundaries.append((line_start, char_pos, 'table_row'))
        
        return boundaries
    
    def token_aware_split(self, text: str, doc_type: DocumentType) -> List[str]:
        """Split text using token-aware chunking with structural respect."""
        if not self.enable_token_aware:
            # Fallback to character-based splitting
            return self._simple_split(text)
        
        config = self.chunk_configs[doc_type]
        normalized_text = self.normalize_text(text)
        boundaries = self.find_structural_boundaries(normalized_text)
        
        chunks = []
        current_pos = 0
        
        while current_pos < len(normalized_text):
            # Calculate target end position
            target_tokens = config.target_tokens
            estimated_chars = target_tokens * 4  # Rough estimation
            target_end = min(current_pos + estimated_chars, len(normalized_text))
            
            # Find a good boundary within the allowed window
            chunk_text = normalized_text[current_pos:target_end]
            actual_tokens = self.count_tokens(chunk_text)
            
            # Adjust if we're over the token limit
            while actual_tokens > config.max_tokens and target_end > current_pos + 100:
                target_end = int(target_end * 0.9)
                chunk_text = normalized_text[current_pos:target_end]
                actual_tokens = self.count_tokens(chunk_text)
            
            # Try to find a good boundary (sentence end, paragraph break, etc.)
            final_end = self._find_optimal_boundary(
                normalized_text, current_pos, target_end, config.boundary_window
            )
            
            # Respect structural boundaries
            final_end = self._respect_structural_boundaries(
                boundaries, current_pos, final_end, normalized_text
            )
            
            chunk_text = normalized_text[current_pos:final_end].strip()
            if chunk_text:
                chunks.append(chunk_text)
            
            # Calculate overlap for next chunk
            overlap_tokens = int(config.target_tokens * config.overlap_percentage / 100)
            overlap_chars = overlap_tokens * 4  # Rough estimation
            
            # For tables, ensure we include headers in overlapping chunks
            if doc_type == DocumentType.TABLES:
                overlap_chars = max(overlap_chars, self._find_table_header_length(chunk_text))
            
            current_pos = max(final_end - overlap_chars, current_pos + 1)
        
        return chunks
    
    def _find_optimal_boundary(self, text: str, start: int, target_end: int, window: int) -> int:
        """Find the optimal boundary within the allowed window."""
        window_start = max(start, target_end - window)
        window_end = min(len(text), target_end + window)
        
        # Look for sentence boundaries first
        for i in range(target_end, window_end):
            if text[i:i+2] == '. ' and i+2 < len(text) and text[i+2].isupper():
                return i + 1
        
        for i in range(target_end, window_start, -1):
            if text[i:i+2] == '. ' and i+2 < len(text) and text[i+2].isupper():
                return i + 1
        
        # Look for paragraph boundaries
        for i in range(target_end, window_end):
            if text[i:i+2] == '\n\n':
                return i + 2
        
        for i in range(target_end, window_start, -1):
            if text[i:i+2] == '\n\n':
                return i + 2
        
        # Look for any line break
        for i in range(target_end, window_end):
            if text[i] == '\n':
                return i + 1
        
        for i in range(target_end, window_start, -1):
            if text[i] == '\n':
                return i + 1
        
        # If no good boundary found, use target_end
        return target_end
    
    def _respect_structural_boundaries(self, boundaries: List[Tuple], start: int, target_end: int, text: str) -> int:
        """Adjust the end position to respect structural boundaries."""
        for boundary_start, boundary_end, boundary_type in boundaries:
            # Don't split within a structural element
            if boundary_start < target_end < boundary_end:
                # If we're close to the start, go before the boundary
                if target_end - boundary_start < boundary_end - target_end:
                    return boundary_start
                else:
                    return boundary_end
            
            # For headings, ensure we don't separate heading from first paragraph
            if boundary_type == 'heading' and boundary_end <= target_end < boundary_end + 200:
                # Find the end of the first paragraph after the heading
                text_after = text[boundary_end:boundary_end + 500]
                paragraph_end = text_after.find('\n\n')
                if paragraph_end != -1:
                    return boundary_end + paragraph_end
        
        return target_end
    
    def _find_table_header_length(self, text: str) -> int:
        """Find the length of table header to include in overlap."""
        lines = text.split('\n')
        header_length = 0
        
        for line in lines:
            if '│' in line or line.count('|') >= 2:
                header_length += len(line) + 1  # +1 for newline
                # Usually first 1-2 rows are headers
                if header_length > 200:  # Reasonable limit
                    break
            else:
                break
        
        return min(header_length, 300)  # Cap at 300 chars
    
    def generate_stable_chunk_id(self, document_id: str, parent_seq: int, child_seq: int) -> str:
        """Generate stable, predictable chunk IDs."""
        return f"{document_id}_{parent_seq:04d}_{child_seq:04d}"
    
    def token_aware_chunking_strategy(self, document: Document) -> List[BaseNode]:
        """Apply token-aware chunking strategy."""
        print(f"Applying token-aware chunking to: {document.metadata.get('file_name', 'Unknown')}")
        
        doc_type = self.classify_document_type(document)
        chunks = self.token_aware_split(document.text, doc_type)
        
        nodes = []
        document_id = document.metadata.get('document_id', document.hash)
        parent_seq = 0
        
        for i, chunk_text in enumerate(chunks):
            chunk_id = self.generate_stable_chunk_id(document_id, parent_seq, i)
            
            node = TextNode(
                text=chunk_text,
                id_=chunk_id,
                metadata={
                    **document.metadata,
                    'chunk_id': chunk_id,
                    'chunk_index': i,
                    'chunk_type': self._classify_chunk_content(chunk_text),
                    'document_type': doc_type.value,
                    'chunking_strategy': 'token_aware',
                    'token_count': self.count_tokens(chunk_text),
                    'parent_sequence': parent_seq,
                    'child_sequence': i
                }
            )
            nodes.append(node)
        
        print(f"Token-aware chunking created {len(nodes)} nodes")
        return nodes
    
    def _classify_chunk_content(self, text: str) -> str:
        """Classify the type of content in a chunk."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['definition', 'means', 'refers to']):
            return 'definition'
        elif any(word in text_lower for word in ['table', 'schedule', 'appendix']):
            return 'table_or_schedule'
        elif any(word in text_lower for word in ['requirement', 'must', 'shall', 'obligation']):
            return 'requirement'
        elif any(word in text_lower for word in ['prohibition', 'prohibited', 'not permitted', 'forbidden']):
            return 'prohibition'
        elif any(word in text_lower for word in ['penalty', 'fine', 'sanction', 'enforcement']):
            return 'enforcement'
        elif re.match(r'^\d+\.\s+', text.strip()):
            return 'section_header'
        else:
            return 'general_content'

def run_ingestion():
    """
    Connects to Pinecone, loads documents, creates an index with improved chunking, and upserts the data.
    """
    index_name = "compliance-bot-index"
    pinecone_dimension = 1536 # Explicitly define dimension

    print("Connecting to Pinecone...")
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY must be set.")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    print(f"Checking if index '{index_name}' exists...")
    if index_name not in pc.list_indexes().names():
        print(f"Index '{index_name}' does not exist. Creating new index...")
        pc.create_index(
            name=index_name,
            dimension=pinecone_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        # Wait for index to be ready
        print("Waiting for index to be ready...")
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        print("Index created successfully.")
    else:
        print(f"Index '{index_name}' already exists.")

    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    
    # Initialize embedding model for chunking and indexing
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-large", 
        api_key=OPENAI_API_KEY,
        dimensions=pinecone_dimension 
    )
    
    # Use UnstructuredReader for parsing PDFs, which is better for tables and images
    unstructured_reader = UnstructuredReader()
    file_extractor = {".pdf": unstructured_reader}

    print("Loading documents from DIFC and ADGM directories...")
    difc_docs = SimpleDirectoryReader(
        "./data/difc", file_extractor=file_extractor
    ).load_data()
    adgm_docs = SimpleDirectoryReader(
        "./data/adgm", file_extractor=file_extractor
    ).load_data()

    # Add jurisdiction metadata to each document
    for doc in difc_docs:
        doc.metadata["jurisdiction"] = "DIFC"
    for doc in adgm_docs:
        doc.metadata["jurisdiction"] = "ADGM"

    # Combine documents from both jurisdictions
    documents = difc_docs + adgm_docs

    # Add a document ID to each document from its hash
    for doc in documents:
        doc.id_ = doc.hash
        doc.metadata["document_id"] = doc.hash

    if not documents:
        print("No documents found in the specified directories. Exiting.")
        return
    
    print(f"Loaded {len(documents)} documents from DIFC and ADGM directories.")

    print("Configuring LlamaIndex settings with advanced token-aware chunking...")
    
    # Initialize the enhanced chunker with token-aware capabilities
    chunker = RegulatoryDocumentChunker(
        embedding_model=embed_model,
        chunk_size=1024,  # Fallback for non-token-aware
        chunk_overlap=128,  # Fallback for non-token-aware
        enable_late_chunking=ENHANCED_MODE,  # Enable if dependencies available
        enable_token_aware=True  # Enable token-aware chunking
    )
    
    # Process documents through advanced chunking
    all_nodes = []
    print("Processing documents with token-aware intelligent chunking...")
    
    for i, doc in enumerate(documents):
        print(f"Processing document {i+1}/{len(documents)}: {doc.metadata.get('file_name', 'Unknown')}")
        try:
            # Use token-aware chunking strategy by default
            nodes = chunker.chunk_document(doc, ChunkingStrategy.TOKEN_AWARE)
            
            # Add enhanced metadata to chunks (with size limits)
            for j, node in enumerate(nodes):
                chunk_metadata = {
                    'chunk_id': f"{doc.metadata.get('document_id', 'unknown')}_{j}",
                    'chunk_index': j,
                    'total_chunks': len(nodes),
                    'chunk_size': len(node.text),
                    'chunk_type': chunker._classify_chunk_content(node.text)
                }
                
                # Only add essential metadata to avoid size limits
                essential_doc_metadata = {
                    'file_name': doc.metadata.get('file_name', 'unknown')[:100],  # Limit length
                    'jurisdiction': doc.metadata.get('jurisdiction', 'unknown'),
                    'document_type': doc.metadata.get('document_type', 'other'),
                    'regulatory_concepts': doc.metadata.get('regulatory_concepts', [])[:3]  # Limit to 3
                }
                
                node.metadata.update({**essential_doc_metadata, **chunk_metadata})
            
            all_nodes.extend(nodes)
            print(f"  Created {len(nodes)} chunks using {nodes[0].metadata.get('chunking_strategy', 'hybrid') if nodes else 'unknown'} strategy")
        except Exception as e:
            print(f"  Error processing document: {e}")
            # Fallback to basic chunking
            basic_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=128)
            nodes = basic_splitter.get_nodes_from_documents([doc])
            for j, node in enumerate(nodes):
                node.metadata.update({
                    'chunk_id': f"{doc.metadata.get('document_id', 'unknown')}_{j}",
                    'chunk_index': j,
                    'total_chunks': len(nodes),
                    'chunk_size': len(node.text),
                    'chunk_type': 'general_content',
                    'chunking_strategy': 'sentence_fallback'
                })
            all_nodes.extend(nodes)
            print(f"  Fallback: Created {len(nodes)} basic chunks")
    
    print(f"Total chunks created: {len(all_nodes)}")
    
    # Analyze chunk distribution
    chunk_types = {}
    chunk_sizes = []
    for node in all_nodes:
        chunk_type = node.metadata.get('chunk_type', 'unknown')
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        chunk_sizes.append(len(node.text))
    
    print("Chunk analysis:")
    print(f"  Average chunk size: {sum(chunk_sizes) / len(chunk_sizes):.0f} characters")
    print(f"  Chunk type distribution: {chunk_types}")

    # Set up LlamaIndex settings
    Settings.embed_model = embed_model

    print("Creating storage context and upserting vectors to Pinecone...")
    try:
        # Create index from nodes instead of documents for better control
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(
            nodes=all_nodes, 
            storage_context=storage_context, 
            show_progress=True
        )
        print("Ingestion complete!")

        # Verification step
        stats = pinecone_index.describe_index_stats()
        print(f"Pinecone index stats: {stats}")
        if stats.total_vector_count > 0:
            print("Successfully verified that vectors were added to the index.")
        else:
            print("Warning: Ingestion complete but no vectors found in the index. Check for issues.")
    except Exception as e:
        print(f"An error occurred during indexing: {e}")


def test_chunking_strategy(sample_docs: int = 3):
    """
    Test the chunking strategy on a sample of documents without uploading to Pinecone.
    """
    print(f"Testing chunking strategy on {sample_docs} sample documents...")
    
    # Initialize embedding model
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-large", 
        api_key=OPENAI_API_KEY,
        dimensions=1536 
    )
    
    # Load a small sample of documents
    unstructured_reader = UnstructuredReader()
    file_extractor = {".pdf": unstructured_reader}
    
    docs = []
    try:
        difc_docs = SimpleDirectoryReader(
            "../data/difc", file_extractor=file_extractor
        ).load_data()[:sample_docs//2]
        adgm_docs = SimpleDirectoryReader(
            "../data/adgm", file_extractor=file_extractor
        ).load_data()[:sample_docs//2]
        docs = difc_docs + adgm_docs
    except Exception as e:
        print(f"Error loading documents: {e}")
        return
    
    if not docs:
        print("No documents found for testing.")
        return
    
    # Initialize chunker with token-aware capabilities
    chunker = RegulatoryDocumentChunker(
        embedding_model=embed_model,
        chunk_size=1024,
        chunk_overlap=128,
        enable_token_aware=True
    )
    
    print(f"\nTesting on {len(docs)} documents:")
    total_chunks = 0
    chunk_stats = {}
    
    for i, doc in enumerate(docs):
        print(f"\n--- Document {i+1}: {doc.metadata.get('file_name', 'Unknown')} ---")
        print(f"Document length: {len(doc.text)} characters")
        
        try:
            nodes = chunker.chunk_document(doc)
            total_chunks += len(nodes)
            
            print(f"Generated {len(nodes)} chunks")
            print(f"Document metadata: {doc.metadata}")
            
            # Analyze chunk types
            for node in nodes:
                chunk_type = node.metadata.get('chunk_type', 'unknown')
                chunk_stats[chunk_type] = chunk_stats.get(chunk_type, 0) + 1
            
            # Show sample chunks
            print("Sample chunks:")
            for j, node in enumerate(nodes[:3]):  # Show first 3 chunks
                print(f"  Chunk {j+1} ({node.metadata.get('chunk_type', 'unknown')}): "
                      f"{node.text[:150]}...")
                
        except Exception as e:
            print(f"Error processing document: {e}")
    
    print(f"\n=== Summary ===")
    print(f"Total chunks generated: {total_chunks}")
    print(f"Average chunks per document: {total_chunks/len(docs):.1f}")
    print(f"Chunk type distribution: {chunk_stats}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_chunking_strategy()
    else:
        run_ingestion()