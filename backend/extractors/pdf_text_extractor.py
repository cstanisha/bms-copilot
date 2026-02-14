"""
Enhanced PDF Text Extraction using PDFMiner
Preserves layout, detects formulas, and extracts structured content
"""
# Standard library imports
import io
import re
from typing import Dict, List, Tuple

# Third-party imports
import pdfplumber
from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import (
    LAParams,
    LTChar,
    LTFigure,
    LTImage,
    LTTextBox,
    LTTextLine,
)



class PDFTextExtractor:
    """
    Advanced PDF text extraction with layout preservation and formula detection
    """
    
    def __init__(self):
        self.laparams = LAParams(
            line_margin=0.5,
            word_margin=0.1,
            char_margin=2.0,
            boxes_flow=0.5,
            detect_vertical=True
        )
    
    def extract_from_file(self, pdf_path: str) -> Dict:
        """
        Extract text and metadata from PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary containing extracted content and metadata
        """
        result = {
            'pages': [],
            'full_text': '',
            'metadata': {},
            'formulas': [],
            'tables': []
        }
        
        # Extract text with layout
        text_by_page = self._extract_text_by_page(pdf_path)
        
        # Extract tables using pdfplumber
        tables_by_page = self._extract_tables(pdf_path)
        
        # Combine results
        for page_num, text in text_by_page.items():
            page_data = {
                'page_number': page_num,
                'text': text,
                'formulas': self._detect_formulas(text),
                'tables': tables_by_page.get(page_num, [])
            }
            result['pages'].append(page_data)
            result['full_text'] += f"\n\n--- Page {page_num} ---\n\n{text}"
        
        return result
    
    def extract_from_uploaded_file(self, uploaded_file) -> Dict:
        """
        Extract text from Streamlit uploaded file object
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            Dictionary containing extracted content and metadata
        """
        # Read file into bytes
        pdf_bytes = uploaded_file.read()
        
        result = {
            'filename': uploaded_file.name,
            'pages': [],
            'full_text': '',
            'metadata': {},
            'formulas': [],
            'tables': []
        }
        
        # Extract text using PDFMiner
        text = extract_text(io.BytesIO(pdf_bytes), laparams=self.laparams)
        result['full_text'] = text
        
        # Extract page-by-page with pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                
                # Extract tables from this page
                tables = page.extract_tables()
                formatted_tables = self._format_tables(tables) if tables else []
                
                # Detect formulas
                formulas = self._detect_formulas(page_text)
                
                page_data = {
                    'page_number': page_num,
                    'text': page_text,
                    'formulas': formulas,
                    'tables': formatted_tables,
                    'has_images': self._has_images(page)
                }
                
                result['pages'].append(page_data)
        
        return result
    
    def _extract_text_by_page(self, pdf_path: str) -> Dict[int, str]:
        """Extract text organized by page"""
        pages = {}
        
        for page_num, page_layout in enumerate(extract_pages(pdf_path, laparams=self.laparams), start=1):
            page_text = []
            
            for element in page_layout:
                if isinstance(element, (LTTextBox, LTTextLine)):
                    text = element.get_text().strip()
                    if text:
                        page_text.append(text)
            
            pages[page_num] = '\n'.join(page_text)
        
        return pages
    
    def _extract_tables(self, pdf_path: str) -> Dict[int, List[Dict]]:
        """Extract tables using pdfplumber"""
        tables_by_page = {}
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                if tables:
                    tables_by_page[page_num] = self._format_tables(tables)
        
        return tables_by_page
    
    def _format_tables(self, tables: List) -> List[Dict]:
        """Format extracted tables into structured format"""
        formatted = []
        
        for table_idx, table in enumerate(tables):
            if not table or len(table) < 2:
                continue
            
            # First row is usually header
            headers = table[0]
            rows = table[1:]
            
            formatted_table = {
                'table_id': table_idx + 1,
                'headers': headers,
                'rows': rows,
                'markdown': self._table_to_markdown(headers, rows)
            }
            formatted.append(formatted_table)
        
        return formatted
    
    def _table_to_markdown(self, headers: List, rows: List[List]) -> str:
        """Convert table to markdown format"""
        if not headers or not rows:
            return ""
        
        # Clean None values
        headers = [str(h) if h else "" for h in headers]
        
        markdown = "| " + " | ".join(headers) + " |\n"
        markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        
        for row in rows:
            clean_row = [str(cell) if cell else "" for cell in row]
            markdown += "| " + " | ".join(clean_row) + " |\n"
        
        return markdown
    
    def _detect_formulas(self, text: str) -> List[Dict]:
        """
        Detect mathematical formulas in text
        Uses heuristics to identify potential formulas
        """
        formulas = []
        
        # Pattern 1: Equations with = sign and mathematical symbols
        equation_pattern = r'([A-Z][A-Za-z0-9]*\s*=\s*[^.]+[+\-*/^()]+[^.]+)'
        matches = re.finditer(equation_pattern, text)
        
        for match in matches:
            formula_text = match.group(1).strip()
            
            # Check if it looks like a mathematical formula
            if self._is_likely_formula(formula_text):
                formulas.append({
                    'text': formula_text,
                    'type': 'equation',
                    'position': match.start()
                })
        
        # Pattern 2: Greek letters (common in formulas)
        greek_pattern = r'(α|β|γ|δ|ε|ζ|η|θ|λ|μ|π|ρ|σ|τ|φ|χ|ψ|ω)'
        if re.search(greek_pattern, text):
            # Context around Greek letters might be formulas
            for match in re.finditer(r'.{0,50}' + greek_pattern + r'.{0,50}', text):
                context = match.group(0).strip()
                if self._is_likely_formula(context):
                    formulas.append({
                        'text': context,
                        'type': 'greek_notation',
                        'position': match.start()
                    })
        
        return formulas
    
    def _is_likely_formula(self, text: str) -> bool:
        """
        Heuristic to determine if text is likely a mathematical formula
        """
        # Must contain mathematical operators
        has_operators = bool(re.search(r'[+\-*/=^()]', text))
        
        # Should have numbers or variables
        has_numbers_or_vars = bool(re.search(r'[0-9A-Za-z]', text))
        
        # Should be relatively short (formulas are usually concise)
        reasonable_length = len(text) < 300
        
        # Should have some mathematical symbols
        math_symbol_count = len(re.findall(r'[+\-*/=^()²³]', text))
        
        return (has_operators and has_numbers_or_vars and 
                reasonable_length and math_symbol_count >= 2)
    
    def _has_images(self, page) -> bool:
        """Check if page contains images"""
        try:
            return len(page.images) > 0
        except:
            return False


# Example usage
if __name__ == "__main__":
    extractor = PDFTextExtractor()
    # result = extractor.extract_from_file("sample.pdf")
    # print(f"Extracted {len(result['pages'])} pages")
    # print(f"Found {len(result['formulas'])} formulas")