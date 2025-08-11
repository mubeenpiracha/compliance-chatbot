"""
Regulatory Context Node - Identifies applicable regulatory framework.
"""
import re
from typing import Any, Dict, List
from openai import AsyncOpenAI
from .base_node import BaseNode
from ..models.query_models import ProcessingState, RegulatoryContext


class RegulatoryContextNode(BaseNode):
    """Identifies the applicable regulatory framework and jurisdiction."""
    
    # Jurisdiction indicators
    JURISDICTION_KEYWORDS = {
        'adgm': ['adgm', 'abu dhabi global market', 'al maryah island'],
        'difc': ['difc', 'dubai international financial centre'],
        'uae': ['uae', 'united arab emirates', 'emirates', 'dubai', 'abu dhabi'],
        'international': ['international', 'cross-border', 'offshore']
    }
    
    # Regulatory domain mapping
    DOMAIN_KEYWORDS = {
        'collective_investment': [
            'fund', 'collective investment', 'investment scheme', 'cis',
            'mutual fund', 'hedge fund', 'private equity', 'portfolio'
        ],
        'licensing': [
            'license', 'permit', 'authorization', 'approval', 'registration',
            'financial services permission', 'fsp'
        ],
        'aml_sanctions': [
            'aml', 'anti-money laundering', 'sanctions', 'kyc', 'cdd',
            'customer due diligence', 'suspicious transactions'
        ],
        'conduct_of_business': [
            'marketing', 'client', 'customer', 'conduct', 'advice',
            'advisory', 'suitability', 'best execution'
        ],
        'market_infrastructure': [
            'market', 'trading', 'clearing', 'settlement', 'exchange',
            'multilateral trading facility', 'mtf'
        ],
        'prudential': [
            'capital', 'liquidity', 'risk management', 'governance',
            'prudential', 'financial resources'
        ]
    }
    
    # Document hierarchy (highest to lowest authority)
    DOCUMENT_HIERARCHY = [
        'ADGM Laws',
        'ADGM Regulations', 
        'FSRA Rulebooks',
        'FSRA Guidance',
        'FSRA Circulars',
        'Industry Guidelines'
    ]
    
    def __init__(self, client: AsyncOpenAI):
        super().__init__("regulatory_context")
        self.client = client
    
    async def process(self, state: ProcessingState, **kwargs) -> Dict[str, Any]:
        """Identify regulatory context for the query."""
        
        classification = state.intermediate_results["compliance_classification"]["classification"]
        query = state.intermediate_results["compliance_classification"]["raw_query"]
        
        # Identify jurisdiction
        jurisdiction = self._identify_jurisdiction(query)
        
        # Identify regulatory domains
        domains = self._identify_domains(query, classification.detected_domains)
        
        # Get applicable frameworks
        frameworks = await self._identify_frameworks(query, jurisdiction, domains)
        
        # Determine document priority for search
        document_priority = self._get_document_priority(domains)
        
        context = RegulatoryContext(
            jurisdiction=jurisdiction,
            regulatory_domains=domains,
            document_priority=document_priority,
            applicable_frameworks=frameworks
        )
        
        return {
            "context": context,
            "analysis": {
                "jurisdiction_confidence": self._get_jurisdiction_confidence(query, jurisdiction),
                "domain_analysis": self._analyze_domains(query, domains),
                "framework_rationale": frameworks
            }
        }
    
    def _identify_jurisdiction(self, query: str) -> str:
        """Identify the most relevant jurisdiction."""
        query_lower = query.lower()
        
        jurisdiction_scores = {}
        for jurisdiction, keywords in self.JURISDICTION_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                jurisdiction_scores[jurisdiction] = score
        
        # Default to ADGM if no clear jurisdiction identified
        if not jurisdiction_scores:
            return 'adgm'
        
        return max(jurisdiction_scores, key=jurisdiction_scores.get)
    
    def _identify_domains(self, query: str, detected_domains: List[str]) -> List[str]:
        """Identify relevant regulatory domains."""
        query_lower = query.lower()
        
        # Start with LLM-detected domains
        domains = set(detected_domains)
        
        # Add keyword-based domain detection
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                domains.add(domain)
        
        return list(domains) if domains else ['general']
    
    async def _identify_frameworks(self, query: str, jurisdiction: str, 
                                 domains: List[str]) -> List[str]:
        """Identify specific regulatory frameworks that apply."""
        
        system_prompt = f"""You are an ADGM regulatory expert. Based on the query and identified context, 
        determine which specific regulatory frameworks are most relevant.
        
        Jurisdiction: {jurisdiction}
        Domains: {', '.join(domains)}
        
        Consider these key ADGM frameworks:
        - Financial Services and Markets Regulations (FSMR)
        - Collective Investment Rules (CIR) 
        - Fund Rules (FUND)
        - Conduct of Business Rulebook (COBS)
        - AML Rulebook
        - General Rulebook (GEN)
        - Prudential Rules
        
        Provide up to 3 most relevant frameworks."""
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}"}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            # Extract frameworks from response
            frameworks_text = response.choices[0].message.content
            frameworks = self._parse_frameworks(frameworks_text)
            
            return frameworks
            
        except Exception as e:
            # Fallback based on domain mapping
            return self._fallback_frameworks(domains)
    
    def _parse_frameworks(self, text: str) -> List[str]:
        """Parse frameworks from LLM response."""
        # Look for common framework abbreviations and names
        framework_patterns = [
            r'\b(FSMR|Financial Services and Markets Regulations)\b',
            r'\b(CIR|Collective Investment Rules)\b',
            r'\b(FUND|Fund Rules)\b',
            r'\b(COBS|Conduct of Business)\b',
            r'\b(AML|Anti-Money Laundering)\b',
            r'\b(GEN|General Rulebook)\b',
            r'\b(PRU|Prudential)\b'
        ]
        
        frameworks = []
        for pattern in framework_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                # Extract the clean framework name
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    frameworks.append(match.group(1))
        
        return frameworks[:3]  # Limit to top 3
    
    def _fallback_frameworks(self, domains: List[str]) -> List[str]:
        """Fallback framework identification based on domains."""
        domain_framework_map = {
            'collective_investment': ['CIR', 'FUND'],
            'licensing': ['FSMR', 'GEN'],
            'aml_sanctions': ['AML'],
            'conduct_of_business': ['COBS'],
            'market_infrastructure': ['FSMR'],
            'prudential': ['PRU', 'GEN']
        }
        
        frameworks = []
        for domain in domains:
            if domain in domain_framework_map:
                frameworks.extend(domain_framework_map[domain])
        
        return list(set(frameworks))  # Remove duplicates
    
    def _get_document_priority(self, domains: List[str]) -> List[str]:
        """Determine document search priority based on domains."""
        # Always maintain hierarchy but may adjust based on domain needs
        priority = self.DOCUMENT_HIERARCHY.copy()
        
        # For certain domains, guidance documents become more important
        if 'conduct_of_business' in domains or 'aml_sanctions' in domains:
            # Move guidance higher for practical implementation questions
            guidance_items = [item for item in priority if 'Guidance' in item or 'Circular' in item]
            other_items = [item for item in priority if item not in guidance_items]
            priority = other_items[:3] + guidance_items + other_items[3:]
        
        return priority
    
    def _get_jurisdiction_confidence(self, query: str, jurisdiction: str) -> float:
        """Calculate confidence in jurisdiction identification."""
        if jurisdiction == 'adgm':
            # Check for explicit ADGM mentions
            adgm_mentions = sum(1 for keyword in self.JURISDICTION_KEYWORDS['adgm']
                              if keyword in query.lower())
            return min(0.9, 0.5 + (adgm_mentions * 0.2))  # Default ADGM assumption
        
        # For other jurisdictions, require explicit mention
        relevant_keywords = self.JURISDICTION_KEYWORDS.get(jurisdiction, [])
        mentions = sum(1 for keyword in relevant_keywords if keyword in query.lower())
        return min(1.0, mentions * 0.4)
    
    def _analyze_domains(self, query: str, domains: List[str]) -> Dict[str, float]:
        """Analyze confidence for each identified domain."""
        query_lower = query.lower()
        domain_confidence = {}
        
        for domain in domains:
            if domain in self.DOMAIN_KEYWORDS:
                keywords = self.DOMAIN_KEYWORDS[domain]
                matches = sum(1 for keyword in keywords if keyword in query_lower)
                confidence = min(1.0, matches * 0.3 + 0.1)
                domain_confidence[domain] = confidence
            else:
                domain_confidence[domain] = 0.5  # Default for LLM-detected domains
        
        return domain_confidence
