"""
Step 4: Abstractive Summarization
Generates concise, rephrased summaries from extractive summaries
"""

import json
import numpy as np
from typing import List, Dict, Set, Tuple
from collections import Counter
import re
import os

try:
    import spacy
except ImportError:
    print("Installing spaCy...")
    os.system("pip install spacy")
    import spacy

class AbstractiveSummarizer:
    """
    Abstractive summarization using NLP techniques
    """
    
    def __init__(self):
        """Initialize spaCy model for NLP tasks"""
        print("Loading spaCy model...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✓ spaCy model loaded successfully")
        except OSError:
            print("Downloading spaCy model (en_core_web_sm)...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
            print("✓ spaCy model downloaded and loaded")
    
    def extract_keywords(self, sentences: List[str], top_n: int = 10) -> List[str]:
        """Extract important keywords"""
        text = ' '.join(sentences)
        doc = self.nlp(text.lower())
        
        word_freq = Counter()
        
        for token in doc:
            if (token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN'] and 
                not token.is_stop and 
                not token.is_punct and
                len(token.text) > 2):
                word_freq[token.lemma_] += 1
        
        keywords = [word for word, count in word_freq.most_common(top_n)]
        return keywords
    
    def extract_entities(self, sentences: List[str]) -> Dict[str, List[str]]:
        """Extract named entities"""
        text = ' '.join(sentences)
        doc = self.nlp(text)
        
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],
            'LAW': [],
            'DATE': [],
            'MONEY': [],
        }
        
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
        
        # Remove duplicates
        for key in entities:
            seen = set()
            entities[key] = [x for x in entities[key] 
                           if not (x.lower() in seen or seen.add(x.lower()))]
        
        return entities
    
    def generate_abstractive_summary(self, extractive_sentences: List[str]) -> Dict:
        """
        Improved abstractive summarization.
        Produces a logically flowing 4–5 line paragraph using:
        - POS tagging
        - NER extraction
        - Verb/subject merging
        - Redundancy removal
        """

        if not extractive_sentences:
            return {
                'abstractive_summary': '',
                'keywords': [],
                'entities': {},
                'key_phrases': []
            }

        # -----------------------------
        # 1. Extract keywords & entities
        # -----------------------------
        keywords = self.extract_keywords(extractive_sentences, top_n=12)
        entities = self.extract_entities(extractive_sentences)

        # -----------------------------
        # 2. Parse sentences with spaCy
        # -----------------------------
        docs = [self.nlp(sent) for sent in extractive_sentences]

        subjects = []
        verbs = []
        objects = []
        modifiers = []

        for doc in docs:
            for token in doc:
                if token.dep_ in ['nsubj', 'nsubjpass'] and token.pos_ in ['NOUN', 'PROPN']:
                    subjects.append(token.text)

                if token.pos_ == 'VERB':
                    verbs.append(token.lemma_)

                if token.dep_ in ['dobj', 'pobj'] and token.pos_ in ['NOUN', 'PROPN']:
                    objects.append(token.text)

                if token.pos_ == 'ADJ':
                    modifiers.append(token.text)

        # Deduplicate but preserve order
        def unique(seq):
            seen = set()
            out = []
            for x in seq:
                if x not in seen:
                    out.append(x)
                    seen.add(x)
            return out

        subjects = unique(subjects)
        verbs = unique(verbs)
        objects = unique(objects)
        modifiers = unique(modifiers)

        # -----------------------------
        # 3. Build content blocks
        # -----------------------------
        # Main actors
        main_subject = subjects[0] if subjects else None

        # Main action
        main_verb = verbs[0] if verbs else "involved"

        # Main object
        main_object = objects[0] if objects else "the case"

        # Combine modifiers into a short descriptor phrase
        description = ", ".join(modifiers[:3]) if modifiers else ""

        # -----------------------------
        # 4. Build a smooth 4–5 line abstractive summary
        # -----------------------------
        summary_lines = []

        # LINE 1: What the case is about
        summary_lines.append(
            f"The case concerns {main_subject or 'the parties involved'}, who {main_verb} {main_object}."
        )

        # LINE 2: Include major entities
        if any(entities.values()):
            parties = []
            for k in ['PERSON', 'ORG', 'GPE']:
                parties.extend(entities[k][:3])
            if parties:
                summary_lines.append(
                    f"Key individuals or entities mentioned include: {', '.join(parties)}."
                )

        # LINE 3: Capture verbs/actions
        if verbs:
            summary_lines.append(
                f"The narrative highlights actions such as {', '.join(verbs[:4])}, reflecting the core events."
            )

        # LINE 4: Context from extractive summary
        clean_first = extractive_sentences[0].strip()
        summary_lines.append(
            f"The proceedings revolve around the following context: {clean_first}"
        )

        # LINE 5: Descriptor or concluding framing
        if description:
            summary_lines.append(
                f"Overall, the situation is described as {description}, summarizing the essence of the dispute."
            )
        else:
            summary_lines.append(
                "Overall, the summary captures the essential elements of the case."
            )

        abstractive_summary = "\n".join(summary_lines)

        return {
            'abstractive_summary': abstractive_summary,
            'keywords': keywords,
            'entities': entities,
            'key_phrases': []
        }

def main():
    """
    Main execution function for Step 4
    """
    print("=" * 70)
    print("STEP 4: ABSTRACTIVE SUMMARIZATION")
    print("=" * 70)
    
    INPUT_FILE = 'extractive_summaries.json'
    OUTPUT_FILE = 'abstractive_summaries.json'
    
    # Load extractive summaries
    print(f"Loading extractive summaries from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            extractive_data = json.load(f)
        print(f"✓ Loaded {len(extractive_data)} extractive summaries")
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found!")
        print("Please run step3.py first")
        return
    
    # Create abstractive summarizer
    print(f"Initializing abstractive summarizer...")
    try:
        summarizer = AbstractiveSummarizer()
    except Exception as e:
        print(f"Error initializing summarizer: {e}")
        return
    
    # Process all summaries
    results = []
    
    print(f"Generating abstractive summaries...")
    for idx, sample in enumerate(extractive_data):
        extractive_sentences = sample.get('summary_sentences', [])
        
        if not extractive_sentences:
            continue
        
        # Generate abstractive summary
        abstractive_result = summarizer.generate_abstractive_summary(
            extractive_sentences
        )
        
        # Combine with existing data
        result = {
            'case_id': sample['case_id'],
            'original_text': sample.get('input', ''),
            'extractive_summary': sample['extractive_summary'],
            'abstractive_summary': abstractive_result['abstractive_summary'],
            'keywords': abstractive_result['keywords'],
            'entities': abstractive_result['entities']
        }
        
        results.append(result)
    
    print(f"✓ Generated {len(results)} abstractive summaries")
    
    # Save results
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {OUTPUT_FILE}")
    
    # Display sample results
    print("\nSample comparisons:")
    for i in range(min(2, len(results))):
        result = results[i]
        print(f"\nCase {i+1}: {result['case_id']}")
        print(f"Extractive: {result['extractive_summary'][:100]}...")
        print(f"Abstractive: {result['abstractive_summary']}")
        print(f"Keywords: {', '.join(result['keywords'][:5])}")
    
    print("\nStep 4 completed successfully!")

if __name__ == "__main__":
    main()
