"""
Mock vector service for testing purposes.
"""
from typing import List, Dict, Any


class VectorService:
    """Mock vector service for testing the enhanced AI system."""
    
    def __init__(self):
        self.mock_documents = [
            {
                'id': 'cir_fund_definition',
                'content': 'A Fund means a Collective Investment Fund. A Collective Investment Fund means any form of collective investment by which the public are invited or may apply to invest money or other property in a portfolio and in relation to which the contributors and the general partners, trustees or operators pool the contributed money or other property and manage it as a whole for the contributors.',
                'score': 0.85,
                'metadata': {
                    'title': 'Collective Investment Rules',
                    'document_type': 'rulebook',
                    'section': 'Definitions',
                    'authority_level': 3,
                    'jurisdiction': 'adgm'
                }
            },
            {
                'id': 'spv_guidance',
                'content': 'Special Purpose Vehicles (SPVs) are legal entities created for a specific purpose, often to isolate financial risk. In the context of investment structures, SPVs may be used to hold specific assets or investments.',
                'score': 0.78,
                'metadata': {
                    'title': 'Investment Structure Guidance',
                    'document_type': 'guidance',
                    'section': 'SPV Structures',
                    'authority_level': 4,
                    'jurisdiction': 'adgm'
                }
            }
        ]
    
    async def search(self, query_vector: List[float], top_k: int = 10, filter_params: Dict = None) -> List[Dict[str, Any]]:
        """Mock vector search that returns sample results."""
        return self.mock_documents[:top_k]
    
    def get_embedding(self, text: str) -> List[float]:
        """Mock embedding generation."""
        # Return a mock embedding vector
        return [0.1] * 1536  # Standard OpenAI embedding size
