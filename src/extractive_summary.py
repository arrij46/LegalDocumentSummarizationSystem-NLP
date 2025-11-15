"""
Step 3: Extractive Summarization
Uses sentence embeddings to extract most important sentences
Based on cosine similarity with document embedding
"""

import json
import numpy as np
from typing import List, Dict, Tuple
import re
import pickle
import os

class ExtractiveSummarizer:
    """
    Extractive summarization using sentence embeddings and cosine similarity
    """
    
    def __init__(self, nlm):
        """
        Initialize with trained Neural Language Model
        """
        self.nlm = nlm
    
    def summarize(self, sentences: List[str], top_k: int = 3) -> Dict:
        """
        Extract top-k most important sentences
        """
        if not sentences:
            return {
                'summary': [],
                'scores': [],
                'indices': [],
                'all_scores': []
            }
        
        # Get sentence embeddings
        sentence_embeddings = []
        valid_sentences = []
        valid_indices = []
        
        for idx, sentence in enumerate(sentences):
            embedding = self.nlm.get_sentence_embedding(sentence)
            
            # Check if embedding is non-zero (valid)
            if np.any(embedding) and np.linalg.norm(embedding) > 0:
                sentence_embeddings.append(embedding)
                valid_sentences.append(sentence)
                valid_indices.append(idx)
        
        if not sentence_embeddings:
            return {
                'summary': [],
                'scores': [],
                'indices': [],
                'all_scores': []
            }
        
        # Convert to numpy array
        sentence_embeddings = np.array(sentence_embeddings)
        
        # Get document embedding (mean of all sentence embeddings)
        document_embedding = np.mean(sentence_embeddings, axis=0)
        
        # Compute cosine similarity for each sentence
        scores = []
        for embedding in sentence_embeddings:
            similarity = self.nlm.cosine_similarity(embedding, document_embedding)
            scores.append(similarity)
        
        # Rank sentences by score (descending order)
        ranked_indices = np.argsort(scores)[::-1]
        
        # Select top-k sentences
        top_k = min(top_k, len(valid_sentences))
        selected_indices = sorted(ranked_indices[:top_k])  # Sort to maintain original order
        
        summary_sentences = [valid_sentences[i] for i in selected_indices]
        summary_scores = [scores[i] for i in selected_indices]
        original_indices = [valid_indices[i] for i in selected_indices]
        
        return {
            'summary': summary_sentences,
            'scores': summary_scores,
            'indices': original_indices,
            'all_scores': scores
        }
    
    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        return sentences

def main():
    """
    Main execution function for Step 3
    """
    print("=" * 70)
    print("STEP 3: EXTRACTIVE SUMMARIZATION")
    print("=" * 70)
    
    # Configuration
    TOP_K = 3
    DATASET_FILE = 'combined_legal_dataset.json'
    MODEL_FILE = 'nlm_model.pkl'
    TOKENIZER_FILE = 'tokenizer.pkl'
    OUTPUT_FILE = 'extractive_summaries.json'
    
    # Load dataset
    print("\nLoading dataset...")
    try:
        with open(DATASET_FILE, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} legal cases")
    except FileNotFoundError:
        print(f"Error: {DATASET_FILE} not found!")
        print("Please run step1.py first")
        return
    
    # Load trained model
    print("\nLoading trained neural language model...")
    try:
        # Import here to avoid circular imports
        from step2 import NeuralLanguageModel
        nlm = NeuralLanguageModel.load(MODEL_FILE, TOKENIZER_FILE)
        print(f"Model loaded successfully")
    except FileNotFoundError:
        print(f"Error: Model files not found!")
        print("Please run step2.py first")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create summarizer
    print("\nInitializing extractive summarizer...")
    summarizer = ExtractiveSummarizer(nlm)
    
    # Generate training data with extractive summaries
    training_data = []
    
    print("\nGenerating extractive summaries...")
    for idx, record in enumerate(dataset):
        sentences = []
        
        # Get sentences from record
        if 'sentences' in record and record['sentences']:
            sentences = record['sentences']
        elif 'ocr_text' in record:
            text = record['ocr_text']
            sentences = ExtractiveSummarizer.split_sentences(text)
        else:
            continue
        
        if not sentences or len(sentences) < 2:
            continue
        
        # Generate extractive summary
        result = summarizer.summarize(sentences, top_k=TOP_K)
        
        if not result['summary']:
            continue
        
        # Create training sample
        sample = {
            'case_id': record.get('case_id', f'case_{idx}'),
            'input': record.get('ocr_text', ' '.join(sentences)),
            'sentences': sentences,
            'extractive_summary': ' '.join(result['summary']),
            'summary_sentences': result['summary'],
            'sentence_scores': result['scores'],
            'selected_indices': result['indices'],
            'num_sentences': len(sentences),
            'num_summary_sentences': len(result['summary'])
        }
        
        training_data.append(sample)
    
    print(f"\nGenerated {len(training_data)} extractive summaries")
    
    # Save training data
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"Training data saved to: {OUTPUT_FILE}")
    
    # Display sample results
    print("\nSample summaries:")
    for i in range(min(2, len(training_data))):
        sample = training_data[i]
        print(f"\nCase {i+1}: {sample['case_id']}")
        print(f"Original sentences: {len(sample['sentences'])}")
        print(f"Summary sentences: {len(sample['summary_sentences'])}")
        print("Summary:")
        for j, sent in enumerate(sample['summary_sentences'], 1):
            print(f"  {j}. {sent[:100]}...")
    
    print("\nStep 3 completed successfully!")

if __name__ == "__main__":
    main()
