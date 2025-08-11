"""
Document corpus loader that reads from the content store.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any
import re

logger = logging.getLogger(__name__)

def load_document_corpus_from_content_store(content_store_path: str = "./content_store") -> List[Dict[str, Any]]:
    """Load document corpus from the content store for keyword search."""
    content_store = Path(content_store_path)
    
    if not content_store.exists():
        logger.warning(f"Content store not found at {content_store_path}")
        return []
    
    documents = []
    
    # Iterate through all document directories
    for doc_dir in content_store.iterdir():
        if not doc_dir.is_dir():
            continue
            
        logger.info(f"Loading documents from {doc_dir.name}")
        
        # Process each chunk file in the directory
        chunk_files = list(doc_dir.glob("*.txt"))
        
        for chunk_file in chunk_files:
            try:
                # Parse filename to extract metadata
                filename_parts = chunk_file.stem.split('_')
                chunk_index = int(filename_parts[0]) if filename_parts[0].isdigit() else 0
                checksum = filename_parts[1] if len(filename_parts) > 1 else ''
                
                # Load content
                content = chunk_file.read_text(encoding='utf-8')
                
                if not content.strip():
                    continue
                
                # Extract metadata from directory name and content
                jurisdiction = _extract_jurisdiction(doc_dir.name)
                document_type = _extract_document_type(doc_dir.name)
                domains = _extract_regulatory_domains(doc_dir.name, content)
                
                document = {
                    'id': f"{doc_dir.name}_{chunk_index:06d}_{checksum}",
                    'content': content,
                    'metadata': {
                        'title': _format_title(doc_dir.name),
                        'document_type': document_type,
                        'section': _extract_section_from_content(content),
                        'authority_level': _determine_authority_level(document_type),
                        'jurisdiction': jurisdiction,
                        'chunk_index': chunk_index,
                        'checksum': checksum,
                        'source_collection': doc_dir.name,
                        'domains': domains,
                        'file_path': str(chunk_file)
                    }
                }
                
                documents.append(document)
                
            except Exception as e:
                logger.error(f"Error processing {chunk_file}: {str(e)}")
                continue
    
    logger.info(f"Loaded {len(documents)} document chunks from content store")
    return documents


def _extract_jurisdiction(dir_name: str) -> str:
    """Extract jurisdiction from directory name."""
    name_lower = dir_name.lower()
    
    # DIFC indicators
    if any(indicator in name_lower for indicator in ['difc', 'dfsa']):
        return 'DIFC'
    
    # ADGM indicators (default for most documents)
    return 'ADGM'


def _extract_document_type(dir_name: str) -> str:
    """Extract document type from directory name."""
    name_lower = dir_name.lower()
    
    if 'rulebook' in name_lower:
        return 'rulebook'
    elif 'regulation' in name_lower:
        return 'regulation'  
    elif 'guidance' in name_lower:
        return 'guidance'
    elif 'law' in name_lower:
        return 'law'
    elif 'rule' in name_lower:
        return 'rules'
    else:
        return 'document'


def _format_title(dir_name: str) -> str:
    """Format directory name into a readable title."""
    # Remove common suffixes and clean up
    title = dir_name.replace('-', ' ').replace('_', ' ')
    title = re.sub(r'-\w+$', '', title)  # Remove trailing codes
    
    # Capitalize words
    words = []
    for word in title.split():
        if word.upper() in ['AML', 'CIR', 'COB', 'DIFC', 'ADGM', 'FSRA', 'DFSA', 'CDD', 'KYC']:
            words.append(word.upper())
        elif len(word) <= 3 and word.lower() in ['and', 'of', 'for', 'in', 'on', 'the']:
            words.append(word.lower())
        else:
            words.append(word.capitalize())
    
    return ' '.join(words)


def _determine_authority_level(document_type: str) -> int:
    """Determine authority level based on document type."""
    authority_map = {
        'law': 1,          # Highest authority
        'regulation': 2,    # High authority  
        'rulebook': 3,     # Medium-high authority
        'rules': 3,        # Medium-high authority
        'guidance': 4,     # Lower authority
        'document': 5      # Lowest authority
    }
    
    return authority_map.get(document_type, 5)


def _extract_regulatory_domains(dir_name: str, content: str) -> List[str]:
    """Extract regulatory domains from directory name and content."""
    domains = []
    name_and_content = (dir_name + ' ' + content[:500]).lower()
    
    # Domain detection patterns
    domain_patterns = {
        'collective_investment': ['collective investment', 'fund', 'cir', 'investment fund'],
        'conduct_of_business': ['conduct of business', 'cob', 'client', 'marketing'],
        'anti_money_laundering': ['anti money laundering', 'aml', 'sanctions', 'suspicious'],
        'prudential': ['prudential', 'capital', 'solvency', 'pin'],
        'market_conduct': ['market', 'trading', 'mkt', 'manipulation'],
        'authorization': ['authorization', 'license', 'permit', 'authorised'],
        'corporate_structure': ['company', 'corporate', 'entity', 'structure'],
        'data_protection': ['data protection', 'privacy', 'personal data'],
        'employment': ['employment', 'employee', 'personnel'],
        'general': ['general', 'gen', 'glossary', 'interpretation']
    }
    
    for domain, patterns in domain_patterns.items():
        if any(pattern in name_and_content for pattern in patterns):
            domains.append(domain)
    
    # Ensure at least one domain
    if not domains:
        domains.append('general')
    
    return domains


def _extract_section_from_content(content: str) -> str:
    """Extract section information from content."""
    lines = content.split('\n')[:10]  # Check first 10 lines
    
    for line in lines:
        line = line.strip()
        # Look for section patterns
        if re.match(r'^\d+\.', line):  # Numbered sections
            return line.split('.')[0] + '.'
        elif re.match(r'^[A-Z]\d+\.', line):  # Like "A1.", "B2."
            return line.split('.')[0] + '.'
        elif re.match(r'^Chapter \d+', line, re.IGNORECASE):  # Chapter references
            return line
        elif re.match(r'^Part \d+', line, re.IGNORECASE):  # Part references
            return line
    
    return 'General'
