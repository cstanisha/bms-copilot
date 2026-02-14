"""
Smart Document Chunker
Preserves context for formulas, tables, and technical content
"""
# Standard library
import re
from typing import Dict, List

# LangChain core imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document



class SmartDocumentChunker:
    """
    Intelligent chunking that preserves important context
    Ensures formulas, tables, and related text stay together
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None
    ):
        if separators is None:
            separators = ["\n\n\n", "\n\n", "\n", ". ", " ", ""]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False
        )
    
    def chunk_document(self, pdf_data: Dict) -> List[Document]:
        """Chunk document while preserving structures"""
        documents = []
        
        for page_data in pdf_data['pages']:
            page_num = page_data['page_number']
            page_text = page_data['text']
            formulas = page_data.get('formulas', [])
            tables = page_data.get('tables', [])
            
            chunks = self._create_contextual_chunks(
                page_text, page_num, formulas, tables,
                pdf_data.get('filename', 'unknown')
            )
            documents.extend(chunks)
        
        return documents
    
    def _create_contextual_chunks(
        self, text: str, page_num: int, formulas: List[Dict],
        tables: List[Dict], filename: str
    ) -> List[Document]:
        """Create chunks with metadata"""
        chunks = self.text_splitter.split_text(text)
        documents = []
        
        for chunk_idx, chunk in enumerate(chunks):
            content_type = self._detect_content_type(chunk, formulas, tables)
            section_info = self._extract_section_info(chunk)
            
            metadata = {
                'source': filename,
                'page': page_num,
                'chunk_id': f"{page_num}_{chunk_idx}",
                'content_type': content_type,
                'has_formula': self._contains_formula(chunk, formulas),
                'has_table': self._contains_table(chunk, tables),
                'section': section_info.get('section', 'unknown'),
                'char_count': len(chunk),
                'word_count': len(chunk.split())
            }
            
            documents.append(Document(page_content=chunk, metadata=metadata))
        
        return documents
    
    def _detect_content_type(self, chunk: str, formulas: List[Dict], tables: List[Dict]) -> str:
        """Detect primary content type"""
        has_formula = self._contains_formula(chunk, formulas)
        has_table = self._contains_table(chunk, tables)
        
        if has_formula and has_table:
            return 'mixed_technical'
        elif has_formula:
            return 'formula_context'
        elif has_table:
            return 'table_data'
        else:
            return 'general_text'
    
    def _contains_formula(self, chunk: str, formulas: List[Dict]) -> bool:
        """Check if chunk contains formulas"""
        return any(f['text'] in chunk for f in formulas)
    
    def _contains_table(self, chunk: str, tables: List[Dict]) -> bool:
        """Check if chunk contains tables"""
        return any(t.get('markdown', '')[:50] in chunk for t in tables)
    
    def _extract_section_info(self, chunk: str) -> Dict:
        """Extract section information"""
        section_pattern = r'(?:Section\s+)?(\d+(?:\.\d+)*)'
        match = re.search(section_pattern, chunk[:200])
        
        return {'section': match.group(1) if match else 'unknown'}