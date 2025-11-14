"""
Step 2: Neural Language Model (NLM) Implementation
Trains word embeddings from scratch using NumPy (CBOW or Skip-gram)
"""

import json
import numpy as np
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import pickle
import os

class Tokenizer:
    def __init__(self, min_freq: int = 2, max_vocab_size: int = 10000):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        
    def tokenize(self, text: str) -> List[str]:
        """Convert text to lowercase tokens"""
        # Lowercase
        text = text.lower()
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts"""
        word_counts = Counter()
        
        for text in texts:
            tokens = self.tokenize(text)
            word_counts.update(tokens)
        
        # Filter by frequency and limit vocab size
        filtered_words = [
            word for word, count in word_counts.most_common(self.max_vocab_size)
            if count >= self.min_freq
        ]
        
        # Add special tokens
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        
        # Add vocabulary
        for idx, word in enumerate(filtered_words, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        self.vocab_size = len(self.word2idx)
        print(f"Vocabulary built: {self.vocab_size} words")
    
    def encode(self, text: str) -> List[int]:
        """Convert text to indices"""
        tokens = self.tokenize(text)
        return [self.word2idx.get(token, 1) for token in tokens]  # 1 is <UNK>
    
    def decode(self, indices: List[int]) -> str:
        """Convert indices back to text"""
        words = [self.idx2word.get(idx, '<UNK>') for idx in indices]
        return ' '.join(words)
    
    def save(self, filepath: str):
        """Save tokenizer"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath: str):
        """Load tokenizer"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

class CBOW_Model:
    """Continuous Bag of Words Model"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 50):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Initialize weights with Xavier initialization
        scale = np.sqrt(2.0 / (vocab_size + embedding_dim))
        self.W1 = np.random.randn(vocab_size, embedding_dim) * scale  # Input to hidden
        self.W2 = np.random.randn(embedding_dim, vocab_size) * scale  # Hidden to output
        
        self.loss_history = []
    
    def softmax(self, x):
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, context_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass
        context_indices: (batch_size, context_size)
        """
        batch_size = len(context_indices)
        context_size = len(context_indices[0])
        
        # One-hot encode context words
        context_one_hot = np.zeros((batch_size, self.vocab_size))
        for i, indices in enumerate(context_indices):
            for idx in indices:
                if 0 <= idx < self.vocab_size:
                    context_one_hot[i, idx] += 1
        
        # Average context embeddings
        context_one_hot /= context_size
        
        # Hidden layer: (batch_size, embedding_dim)
        hidden = np.dot(context_one_hot, self.W1)
        
        # Output layer: (batch_size, vocab_size)
        output = np.dot(hidden, self.W2)
        
        # Apply softmax
        probs = self.softmax(output)
        
        return hidden, probs
    
    def backward(self, context_indices: np.ndarray, target_indices: np.ndarray, 
                 hidden: np.ndarray, probs: np.ndarray, learning_rate: float):
        """
        Backward pass
        """
        batch_size = len(context_indices)
        context_size = len(context_indices[0])
        
        # One-hot encode targets
        target_one_hot = np.zeros((batch_size, self.vocab_size))
        for i, idx in enumerate(target_indices):
            if 0 <= idx < self.vocab_size:
                target_one_hot[i, idx] = 1
        
        # Gradient of loss w.r.t. output
        d_output = probs - target_one_hot  # (batch_size, vocab_size)
        
        # Gradient for W2
        d_W2 = np.dot(hidden.T, d_output) / batch_size
        
        # Gradient for hidden layer
        d_hidden = np.dot(d_output, self.W2.T)
        
        # One-hot encode context
        context_one_hot = np.zeros((batch_size, self.vocab_size))
        for i, indices in enumerate(context_indices):
            for idx in indices:
                if 0 <= idx < self.vocab_size:
                    context_one_hot[i, idx] += 1
        context_one_hot /= context_size
        
        # Gradient for W1
        d_W1 = np.dot(context_one_hot.T, d_hidden) / batch_size
        
        # Update weights
        self.W1 -= learning_rate * d_W1
        self.W2 -= learning_rate * d_W2
    
    def compute_loss(self, probs: np.ndarray, target_indices: np.ndarray) -> float:
        """Cross-entropy loss"""
        batch_size = len(target_indices)
        loss = 0
        for i, idx in enumerate(target_indices):
            if 0 <= idx < self.vocab_size:
                loss -= np.log(probs[i, idx] + 1e-10)
        return loss / batch_size
    
    def get_embedding(self, word_idx: int) -> np.ndarray:
        """Get embedding for a word"""
        if 0 <= word_idx < self.vocab_size:
            return self.W1[word_idx]
        return np.zeros(self.embedding_dim)

class NeuralLanguageModel:
    def __init__(self, embedding_dim: int = 50, window_size: int = 2):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.tokenizer = None
        self.model = None
    
    def prepare_cbow_data(self, token_indices: List[int]) -> Tuple[List, List]:
        """
        Prepare training data for CBOW
        Predict center word from context words
        """
        contexts = []
        targets = []
        
        for i in range(self.window_size, len(token_indices) - self.window_size):
            # Get context words (before and after target)
            context = (
                token_indices[i - self.window_size:i] + 
                token_indices[i + 1:i + self.window_size + 1]
            )
            target = token_indices[i]
            
            contexts.append(context)
            targets.append(target)
        
        return contexts, targets
    
    def train(self, texts: List[str], epochs: int = 5, 
              learning_rate: float = 0.01, batch_size: int = 64):
        """
        Train the neural language model
        """
        print("\n" + "=" * 60)
        print("TRAINING NEURAL LANGUAGE MODEL")
        print("=" * 60)
        
        # Build tokenizer
        print("\nBuilding vocabulary...")
        self.tokenizer = Tokenizer(min_freq=2, max_vocab_size=10000)
        self.tokenizer.build_vocab(texts)
        
        # Initialize model
        self.model = CBOW_Model(self.tokenizer.vocab_size, self.embedding_dim)
        
        # Prepare all training data
        print("\nPreparing training data...")
        all_contexts = []
        all_targets = []
        
        for text in texts:
            token_indices = self.tokenizer.encode(text)
            if len(token_indices) > self.window_size * 2:
                contexts, targets = self.prepare_cbow_data(token_indices)
                all_contexts.extend(contexts)
                all_targets.extend(targets)
        
        print(f"Training samples: {len(all_contexts)}")
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            # Shuffle data
            indices = np.random.permutation(len(all_contexts))
            
            for i in range(0, len(all_contexts), batch_size):
                batch_indices = indices[i:i + batch_size]
                
                batch_contexts = [all_contexts[j] for j in batch_indices]
                batch_targets = [all_targets[j] for j in batch_indices]
                
                # Convert to numpy arrays
                batch_contexts = np.array(batch_contexts)
                batch_targets = np.array(batch_targets)
                
                # Forward pass
                hidden, probs = self.model.forward(batch_contexts)
                
                # Compute loss
                loss = self.model.compute_loss(probs, batch_targets)
                epoch_loss += loss
                num_batches += 1
                
                # Backward pass
                self.model.backward(batch_contexts, batch_targets, 
                                  hidden, probs, learning_rate)
            
            avg_loss = epoch_loss / num_batches
            self.model.loss_history.append(avg_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        print("\n✓ Training completed!")
    
    def get_sentence_embedding(self, sentence: str) -> np.ndarray:
        """
        Get embedding for a sentence (average of word embeddings)
        """
        token_indices = self.tokenizer.encode(sentence)
        
        if not token_indices:
            return np.zeros(self.embedding_dim)
        
        embeddings = [self.model.get_embedding(idx) for idx in token_indices]
        return np.mean(embeddings, axis=0)
    
    def get_document_embedding(self, sentences: List[str]) -> np.ndarray:
        """
        Get embedding for a document (average of sentence embeddings)
        """
        sentence_embeddings = [self.get_sentence_embedding(s) for s in sentences]
        return np.mean(sentence_embeddings, axis=0)
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def save(self, model_path: str = "nlm_model.pkl", 
             tokenizer_path: str = "tokenizer.pkl"):
        """Save model and tokenizer"""
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        self.tokenizer.save(tokenizer_path)
        print(f"\nModel saved to {model_path}")
        print(f"Tokenizer saved to {tokenizer_path}")
    
    @staticmethod
    def load(model_path: str = "nlm_model.pkl", 
             tokenizer_path: str = "tokenizer.pkl"):
        """Load model and tokenizer"""
        nlm = NeuralLanguageModel()
        
        with open(model_path, 'rb') as f:
            nlm.model = pickle.load(f)
        
        nlm.tokenizer = Tokenizer.load(tokenizer_path)
        nlm.embedding_dim = nlm.model.embedding_dim
        
        return nlm

def main():
    """
    Main execution function
    """
    # Load dataset
    print("Loading dataset...")
    try:
        with open('combined_legal_dataset.json', 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} legal cases")
    except FileNotFoundError:
        print("Error: combined_legal_dataset.json not found!")
        print("Please run step1.py first")
        return
    
    # Extract all sentences for training
    all_sentences = []
    for record in dataset:
        if 'sentences' in record:
            all_sentences.extend(record['sentences'])
        elif 'ocr_text' in record:
            # If sentences not pre-split, split them
            text = record['ocr_text']
            sentences = re.split(r'(?<=[.!?])\s+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            all_sentences.extend(sentences)
    
    print(f"Total sentences: {len(all_sentences)}")
    
    if len(all_sentences) < 100:
        print("Warning: Very few sentences for training. Model may not perform well.")
    
    # Train model
    nlm = NeuralLanguageModel(embedding_dim=50, window_size=2)
    nlm.train(all_sentences, epochs=5, learning_rate=0.01, batch_size=64)
    
    # Save model
    nlm.save()
    
    print("\nStep 2 completed successfully!")

if __name__ == "__main__":
    main()
