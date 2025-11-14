"""
Step 1: OCR and Preprocessing Pipeline
Extracts text from legal judgement PDFs and creates structured JSON
"""

import json
import os
import re
from pathlib import Path
from pdf2image import convert_from_path
import pytesseract
from typing import List, Dict
import string

class LegalPDFProcessor:
    def __init__(self, pdf_folder: str, output_folder: str):
        self.pdf_folder = Path(pdf_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Convert PDF to images and extract text using OCR
        """
        try:
            # Convert PDF pages to images
            images = convert_from_path(pdf_path, dpi=300)
            
            extracted_text = []
            for i, image in enumerate(images):
                # Extract text from each page
                text = pytesseract.image_to_string(image, lang='eng')
                extracted_text.append(text)
            
            # Combine all pages
            full_text = "\n\n".join(extracted_text)
            return full_text
        
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix hyphenation (words split across lines)
        text = re.sub(r'-\s+', '', text)
        
        # Remove page numbers (assuming format like "Page 1", "- 2 -", etc.)
        text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'-\s*\d+\s*-', '', text)
        
        # Remove common headers/footers patterns
        text = re.sub(r'IN THE (?:SUPREME|HIGH) COURT.*?\n', '', text, flags=re.IGNORECASE)
        
        # Fix common OCR errors
        text = text.replace('|', 'I')
        text = text.replace('~', '-')
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        """
        # Basic sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter out very short sentences (likely noise)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        return sentences
    
    def process_pdf(self, pdf_path: Path, case_id: str) -> Dict:
        """
        Complete processing pipeline for a single PDF
        """
        print(f"Processing: {pdf_path.name}")
        
        # Extract text
        raw_text = self.extract_text_from_pdf(str(pdf_path))
        
        if not raw_text:
            return None
        
        # Clean text
        cleaned_text = self.clean_text(raw_text)
        
        # Split into sentences
        sentences = self.split_into_sentences(cleaned_text)
        
        # Create record
        record = {
            "case_id": case_id,
            "pdf_source": str(pdf_path),
            "ocr_text": cleaned_text,
            "sentences": sentences,
            "num_sentences": len(sentences),
            "num_words": len(cleaned_text.split())
        }
        
        return record
    
    def process_all_pdfs(self) -> List[Dict]:
        """
        Process all PDFs in the folder
        """
        pdf_files = list(self.pdf_folder.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {self.pdf_folder}")
            return []
        
        print(f"Found {len(pdf_files)} PDF files")
        
        all_records = []
        for idx, pdf_path in enumerate(pdf_files, 1):
            case_id = f"SCP_{2025}_{idx:03d}"
            
            record = self.process_pdf(pdf_path, case_id)
            
            if record:
                all_records.append(record)
                
                # Save individual record
                output_file = self.output_folder / f"{case_id}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(record, f, indent=2, ensure_ascii=False)
                
                print(f"  -> Saved: {output_file.name}")
        
        return all_records
    
    def merge_with_existing(self, new_records: List[Dict], 
                          existing_json_path: str = None) -> List[Dict]:
        """
        Merge OCR records with existing scraped data
        """
        existing_data = []
        
        if existing_json_path and os.path.exists(existing_json_path):
            with open(existing_json_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            print(f"Loaded {len(existing_data)} existing records")
        
        # Create a set of existing case IDs to avoid duplicates
        existing_ids = {record.get('case_id') for record in existing_data}
        
        # Add new records that don't exist
        merged_data = existing_data.copy()
        new_count = 0
        
        for record in new_records:
            if record['case_id'] not in existing_ids:
                merged_data.append(record)
                new_count += 1
        
        print(f"Added {new_count} new records")
        print(f"Total records: {len(merged_data)}")
        
        return merged_data
    
    def save_combined_dataset(self, records: List[Dict], 
                             output_path: str = "combined_legal_dataset.json"):
        """
        Save the final combined dataset
        """
        output_path = Path(output_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        
        print(f"\nCombined dataset saved to: {output_path}")
        print(f"Total cases: {len(records)}")
        
        # Print statistics
        total_sentences = sum(r['num_sentences'] for r in records)
        total_words = sum(r['num_words'] for r in records)
        
        print(f"Total sentences: {total_sentences}")
        print(f"Total words: {total_words}")
        print(f"Average sentences per case: {total_sentences/len(records):.1f}")
        print(f"Average words per case: {total_words/len(records):.1f}")


def main():
    """
    Main execution function
    """
    # Configure paths
    PDF_FOLDER = "supreme_court_judgements"  # Folder containing PDF files
    OUTPUT_FOLDER = "ocr_output"  # Folder for individual JSON files
    EXISTING_JSON = "existing_scraped_data.json"  # Optional: existing dataset
    COMBINED_OUTPUT = "combined_legal_dataset.json"  # Final merged dataset
    
    # Initialize processor
    processor = LegalPDFProcessor(PDF_FOLDER, OUTPUT_FOLDER)
    
    # Process all PDFs
    print("=" * 60)
    print("STEP 1: OCR AND PREPROCESSING")
    print("=" * 60)
    
    new_records = processor.process_all_pdfs()
    
    if not new_records:
        print("No records processed. Please check PDF folder.")
        return
    
    # Merge with existing data (if available)
    print("\n" + "=" * 60)
    print("MERGING WITH EXISTING DATA")
    print("=" * 60)
    
    combined_records = processor.merge_with_existing(
        new_records, 
        EXISTING_JSON if os.path.exists(EXISTING_JSON) else None
    )
    
    # Save combined dataset
    processor.save_combined_dataset(combined_records, COMBINED_OUTPUT)
    
    print("\nStep 1 completed successfully!")


if __name__ == "__main__":
    main()
