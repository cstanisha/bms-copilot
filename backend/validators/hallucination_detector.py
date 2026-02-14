"""
Hallucination Detector
Validates that AI responses only use information from source documents
"""
from typing import Dict, List
from langchain_core.documents import Document
import re


class HallucinationDetector:
    """
    Detects potential hallucinations in AI-generated responses
    Ensures answers are grounded in source documents
    """
    
    def __init__(self, min_source_coverage: float = 0.8):
        """
        Args:
            min_source_coverage: Minimum percentage of answer that must be sourced
        """
        self.min_source_coverage = min_source_coverage
    
    def validate_answer(
        self,
        answer: str,
        source_documents: List[Document],
        query: str
    ) -> Dict:
        """
        Validate that answer is grounded in source documents
        
        Args:
            answer: AI-generated answer
            source_documents: Retrieved source documents
            query: Original user query
            
        Returns:
            Validation result dictionary
        """
        # Extract factual claims from answer
        claims = self._extract_claims(answer)
        
        # Check each claim against sources
        claim_validation = []
        for claim in claims:
            is_supported, evidence = self._verify_claim(claim, source_documents)
            claim_validation.append({
                'claim': claim,
                'supported': is_supported,
                'evidence': evidence
            })
        
        # Calculate coverage
        supported_claims = sum(1 for cv in claim_validation if cv['supported'])
        total_claims = len(claims)
        coverage = supported_claims / total_claims if total_claims > 0 else 0
        
        # Check for external knowledge indicators
        external_knowledge_detected = self._detect_external_knowledge(answer, source_documents)
        
        # Check for proper citations
        has_citations = self._check_citations(answer)
        
        # Overall validation
        is_valid = (
            coverage >= self.min_source_coverage and
            not external_knowledge_detected and
            has_citations
        )
        
        return {
            'valid': is_valid,
            'confidence': coverage,
            'coverage': coverage,
            'total_claims': total_claims,
            'supported_claims': supported_claims,
            'unsupported_claims': [cv for cv in claim_validation if not cv['supported']],
            'external_knowledge_detected': external_knowledge_detected,
            'has_citations': has_citations,
            'recommendation': 'APPROVE' if is_valid else 'REJECT_OR_MODIFY'
        }
    
    def _extract_claims(self, answer: str) -> List[str]:
        """
        Extract factual claims from answer
        Simple sentence-based extraction
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', answer)
        
        # Filter out non-factual sentences (questions, citations, etc.)
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Too short
                continue
            if sentence.startswith('According to'):  # Citation
                continue
            if sentence.startswith('Source:'):  # Citation
                continue
            if '?' in sentence:  # Question
                continue
            
            claims.append(sentence)
        
        return claims
    
    def _verify_claim(
        self,
        claim: str,
        source_documents: List[Document]
    ) -> tuple[bool, str]:
        """
        Verify if claim is supported by source documents
        
        Returns:
            (is_supported, evidence_text)
        """
        claim_lower = claim.lower()
        
        # Check each source document
        for doc in source_documents:
            content_lower = doc.page_content.lower()
            
            # Simple containment check
            # In production, use more sophisticated semantic similarity
            if self._semantic_overlap(claim_lower, content_lower):
                # Found supporting evidence
                evidence = self._extract_relevant_snippet(claim, doc.page_content)
                return True, evidence
        
        return False, ""
    
    def _semantic_overlap(self, claim: str, source: str) -> bool:
        """
        Check if claim has significant overlap with source
        Simple keyword-based approach
        """
        # Extract key terms from claim (nouns, numbers, technical terms)
        claim_terms = set(re.findall(r'\b[A-Z][a-z]+\b|\b\d+(?:\.\d+)?\b', claim))
        
        if not claim_terms:
            # Fallback to all words
            claim_terms = set(claim.split())
        
        # Check how many terms appear in source
        matches = sum(1 for term in claim_terms if term.lower() in source)
        
        # Require at least 50% of terms to match
        overlap_ratio = matches / len(claim_terms) if claim_terms else 0
        return overlap_ratio >= 0.5
    
    def _extract_relevant_snippet(self, claim: str, source: str, window: int = 200) -> str:
        """Extract relevant snippet from source that supports claim"""
        # Find best matching position
        claim_words = claim.lower().split()[:5]  # First 5 words
        
        for word in claim_words:
            pos = source.lower().find(word)
            if pos != -1:
                start = max(0, pos - window // 2)
                end = min(len(source), pos + window // 2)
                return source[start:end]
        
        return source[:window]  # Fallback
    
    def _detect_external_knowledge(self, answer: str, source_documents: List[Document]) -> bool:
        """
        Detect if answer contains information not in sources
        """
        # Common phrases indicating external knowledge
        external_indicators = [
            'in general',
            'typically',
            'usually',
            'commonly',
            'it is well known',
            'as we know',
            'in my experience'
        ]
        
        answer_lower = answer.lower()
        for indicator in external_indicators:
            if indicator in answer_lower:
                return True
        
        return False
    
    def _check_citations(self, answer: str) -> bool:
        """
        Check if answer includes proper citations
        """
        # Look for citation patterns
        citation_patterns = [
            r'page \d+',
            r'section \d+',
            r'according to',
            r'source:',
            r'from page',
            r'equation \d+',
            r'table \d+'
        ]
        
        answer_lower = answer.lower()
        for pattern in citation_patterns:
            if re.search(pattern, answer_lower):
                return True
        
        return False


# Example usage
if __name__ == "__main__":
    detector = HallucinationDetector(min_source_coverage=0.8)
    
    # Example validation
    answer = "According to page 47, the pressure loss formula is P = 4.52 × Q^1.85 / (C^1.85 × d^4.87)"
    sources = []  # Would contain actual documents
    
    result = detector.validate_answer(answer, sources, "What is the pressure loss formula?")
    print(f"Valid: {result['valid']}, Confidence: {result['confidence']}")