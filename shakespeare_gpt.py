
import os
import math
import time
import inspect
from dataclasses import dataclass
from typing import List, Tuple, Optional
import re

import torch
import torch.nn as nn
from torch.nn import functional as F
import requests
import tiktoken

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Model architecture classes (CausalSelfAttention, MLP, Block remain the same)
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, weight_decay, learning_rate):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameters: {num_decay_params:,}")
        print(f"num non-decayed parameters: {num_nodecay_params:,}")

        use_fused = (
            'fused' in inspect.signature(torch.optim.AdamW).parameters
            and torch.cuda.is_available()
            and DEVICE.type == 'cuda'
        )

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            fused=use_fused
        )

        return optimizer

# Dataset class
class ShakespeareDataset:
    def __init__(self):
        try:
            # Try reading from local file first
            if os.path.exists('shakespeare.txt'):
                with open('shakespeare.txt', 'r', encoding='utf-8') as f:
                    self.text = f.read()
                print("Successfully loaded from local file")
            else:
                # Fallback to URL if local file doesn't exist
                shakespeare_url = "https://raw.githubusercontent.com/karpathy/build-nanogpt/master/input.txt"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(shakespeare_url, headers=headers, timeout=30)
                response.raise_for_status()
                self.text = response.text
                
                # Save the file locally for future use
                with open('shakespeare.txt', 'w', encoding='utf-8') as f:
                    f.write(self.text)
                print("Successfully loaded from URL and saved locally")
            
            self.enc = tiktoken.get_encoding("gpt2")
            self.tokens = self.enc.encode(self.text)
            self.character_lines = self._extract_character_lines()
            print(f"Dataset loaded with {len(self.tokens):,} tokens")
            
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            raise
    def _extract_character_lines(self) -> dict:
        lines = self.text.split('\n')
        character_lines = {}
        current_character = None
        
        for line in lines:
            if line.strip().isupper() and len(line.split()) <= 3:
                current_character = line.strip()
                if current_character not in character_lines:
                    character_lines[current_character] = []
            elif current_character and line.strip():
                character_lines[current_character].append(line.strip())
                
        return character_lines
    
    def get_character_list(self) -> List[str]:
        return list(self.character_lines.keys())
    
    def get_character_lines(self, character: str) -> List[str]:
        return self.character_lines.get(character, [])
    
    def search_text(self, query: str) -> List[str]:
        matches = []
        for line in self.text.split('\n'):
            if query.lower() in line.lower():
                matches.append(line.strip())
        return matches
            
class ShakespeareBot:
    def __init__(self, model_path: Optional[str] = None):
        self.dataset = ShakespeareDataset()
        self.device = DEVICE
        
        # Initialize model
        self.model = GPT(GPTConfig(vocab_size=50304))
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
    
    def process_query(self, query: str) -> str:
        """Process user queries about Shakespeare's text."""
        query = query.lower().strip()
        
        if "who is" in query or "tell me about" in query:
            character = self._extract_character_name(query)
            if character:
                return self._get_character_info(character)
        elif "search for" in query or "find" in query:
            search_term = query.split("search for")[-1].strip() if "search for" in query else query.split("find")[-1].strip()
            return self._search_text(search_term)
        elif "generate" in query or "create" in query:
            prompt = query.split("generate")[-1].strip() if "generate" in query else query.split("create")[-1].strip()
            return self._generate_text(prompt)
        elif "list characters" in query:
            return self._list_characters()
        else:
            return self.get_help_message()
    
    def _extract_character_name(self, query: str) -> Optional[str]:
        """Extract character name from query."""
        characters = self.dataset.get_character_list()
        for character in characters:
            if character.lower() in query.lower():
                return character
        return None

    def _get_character_info(self, character: str) -> str:
        lines = self.dataset.get_character_lines(character)
        if not lines:
            return f"Character '{character}' not found."
        
        num_lines = len(lines)
        sample_lines = lines[:3]
        
        response = f"Character: {character}\n"
        response += f"Number of lines: {num_lines}\n"
        response += "Sample lines:\n"
        for line in sample_lines:
            response += f"- {line}\n"
        
        return response

    def _search_text(self, search_term: str) -> str:
        matches = self.dataset.search_text(search_term)
        if not matches:
            return f"No matches found for '{search_term}'"
        
        response = f"Found {len(matches)} matches for '{search_term}':\n"
        for i, match in enumerate(matches[:5], 1):
            response += f"{i}. {match}\n"
        
        if len(matches) > 5:
            response += f"... and {len(matches) - 5} more matches"
        
        return response

    def _list_characters(self) -> str:
        characters = self.dataset.get_character_list()
        response = "Characters in the play:\n"
        for i, character in enumerate(characters, 1):
            response += f"{i}. {character}\n"
        return response

    def _generate_text(self, prompt: str, max_tokens: int = 50) -> str:
        enc = tiktoken.get_encoding("gpt2")
        tokens = torch.tensor(enc.encode(prompt)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for _ in range(max_tokens):
                if tokens.size(1) >= self.model.config.block_size:
                    break
                    
                logits, _ = self.model(tokens)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                
                next_token = torch.multinomial(probs, num_samples=1)
                tokens = torch.cat([tokens, next_token], dim=1)
        
        generated_text = enc.decode(tokens[0].tolist())
        return f"Generated text:\n{generated_text}"

    def get_help_message(self) -> str:
        return """I can help you explore Shakespeare's works! Try:

1. Character Questions:
   - "Who is Hamlet?"
   - "Tell me about Ophelia"

2. Search for Quotes:
   - "Search for to be or not to be"
   - "Find love quotes"

3. Generate Text:
   - "Generate a soliloquy about love"
   - "Create a dialogue"

4. Other Commands:
   - "List characters"

Type your question or 'exit' to quit."""

# Analysis Components
class ShakespeareAnalyzer:
    def __init__(self, dataset: ShakespeareDataset):
        self.dataset = dataset
        self.text = dataset.text
        self.char_lines = dataset.character_lines

    def get_play_statistics(self) -> dict:
        words = self.text.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', self.text) if s.strip()]
        
        stats = {
            'total_words': len(words),
            'total_characters': len(self.char_lines),
            'total_sentences': len(sentences),
            'average_sentence_length': len(words) / len(sentences),
            'unique_words': len(set(words)),
            'longest_speech': self._find_longest_speech(),
            'most_talkative_character': self._find_most_talkative_character(),
        }
        return stats

    def _find_longest_speech(self) -> Tuple[str, str, int]:
        max_length = 0
        longest_speech = ''
        speaker = ''
        
        for char, lines in self.char_lines.items():
            for line in lines:
                if len(line.split()) > max_length:
                    max_length = len(line.split())
                    longest_speech = line
                    speaker = char
        
        return speaker, longest_speech, max_length

    def _find_most_talkative_character(self) -> Tuple[str, int]:
        char_word_counts = {
            char: sum(len(line.split()) for line in lines)
            for char, lines in self.char_lines.items()
        }
        most_talkative = max(char_word_counts.items(), key=lambda x: x[1])
        return most_talkative

    def analyze_character_relationships(self) -> dict:
        relationships = {}
        current_scene_chars = set()
        
        for line in self.text.split('\n'):
            if line.strip().isupper() and len(line.split()) <= 3:
                char = line.strip()
                current_scene_chars.add(char)
            elif line.startswith('Scene') or line.startswith('ACT'):
                if current_scene_chars:
                    for char1 in current_scene_chars:
                        for char2 in current_scene_chars:
                            if char1 < char2:
                                pair = (char1, char2)
                                relationships[pair] = relationships.get(pair, 0) + 1
                current_scene_chars = set()
                
        return relationships

    def get_character_sentiment(self, character: str) -> dict:
        positive_words = {'good', 'love', 'happy', 'joy', 'sweet', 'gentle', 'kind'}
        negative_words = {'bad', 'hate', 'sad', 'anger', 'death', 'cruel', 'bitter'}
        
        lines = self.char_lines.get(character, [])
        words = ' '.join(lines).lower().split()
        
        sentiment = {
            'positive_count': sum(1 for word in words if word in positive_words),
            'negative_count': sum(1 for word in words if word in negative_words),
            'total_words': len(words)
        }
        if sentiment['total_words'] > 0:
            sentiment['sentiment_ratio'] = (
                (sentiment['positive_count'] - sentiment['negative_count']) /
                sentiment['total_words']
            )
        else:
            sentiment['sentiment_ratio'] = 0
        
        return sentiment

# Enhanced Shakespeare Bot
class EnhancedShakespeareBot(ShakespeareBot):
    def __init__(self, model_path: Optional[str] = None):
        super().__init__(model_path)
        self._initialize_character_cache()
        self.analyzer = ShakespeareAnalyzer(self.dataset)
    
    def _initialize_character_cache(self):
        """Pre-process character names for better matching."""
        self.character_cache = {}
        for character in self.dataset.get_character_list():
            normalized = character.lower().strip()
            self.character_cache[normalized] = character
            if ' ' in normalized:
                short_name = normalized.split()[0]
                self.character_cache[short_name] = character

    def _extract_character_name(self, query: str) -> Optional[str]:
        query = query.lower().strip()
        for prefix in ["who is ", "tell me about ", "what about "]:
            if query.startswith(prefix):
                query = query[len(prefix):]
                break
        
        if query in self.character_cache:
            return self.character_cache[query]
        
        for char_variant, full_name in self.character_cache.items():
            if char_variant in query or query in char_variant:
                return full_name
        
        return None

    def process_query(self, query: str) -> str:
        try:
            query = query.lower().strip()
            
            # Check for analytical queries
            if "analyze" in query:
                if "relationships" in query:
                    return self._show_character_relationships()
                elif "sentiment" in query:
                    return self._show_character_sentiments()
                elif "statistics" in query:
                    return self._show_play_statistics()
            
            # Check for specific character analysis
            if "analyze character" in query:
                character = self._extract_character_name(query)
                if character:
                    return self._analyze_character(character)
            
            # Default to parent class processing
            return super().process_query(query)
            
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}\nPlease try rephrasing your question."

    def _show_character_relationships(self) -> str:
        relationships = self.analyzer.analyze_character_relationships()
        result = "Character Relationships:\n\n"
        for (char1, char2), count in sorted(relationships.items(), key=lambda x: x[1], reverse=True)[:10]:
            result += f"{char1} and {char2}: {count} scenes together\n"
        return result

    def _show_character_sentiments(self) -> str:
        result = "Character Sentiment Analysis:\n\n"
        characters = self.dataset.get_character_list()[:10]  # Top 10 characters
        
        for char in characters:
            sentiment = self.analyzer.get_character_sentiment(char)
            bar = "+" * int((sentiment['sentiment_ratio'] + 1) * 20) if sentiment['sentiment_ratio'] > 0 else "-" * int((-sentiment['sentiment_ratio'] + 1) * 20)
            result += f"{char:<20} {bar} ({sentiment['sentiment_ratio']:.2f})\n"
        
        return result

    def _show_play_statistics(self) -> str:
        stats = self.analyzer.get_play_statistics()
        return f"""Play Statistics:
Total Words: {stats['total_words']:,}
Total Characters: {stats['total_characters']}
Total Sentences: {stats['total_sentences']:,}
Average Sentence Length: {stats['average_sentence_length']:.1f} words
Unique Words: {stats['unique_words']:,}
Most Talkative Character: {stats['most_talkative_character'][0]} ({stats['most_talkative_character'][1]:,} words)
"""

    def _analyze_character(self, character: str) -> str:
        lines = self.dataset.get_character_lines(character)
        sentiment = self.analyzer.get_character_sentiment(character)
        
        return f"""Character Analysis: {character}

Speech Statistics:
- Total Lines: {len(lines)}
- Total Words: {sentiment['total_words']:,}
- Sentiment Analysis:
  * Positive References: {sentiment['positive_count']}
  * Negative References: {sentiment['negative_count']}
  * Overall Sentiment: {"Positive" if sentiment['sentiment_ratio'] > 0 else "Negative"} ({sentiment['sentiment_ratio']:.2f})

Sample Quotes:
{chr(10).join(f"- {line}" for line in lines[:3])}
"""

# Training Components
class DataLoaderLite:
    def __init__(self, B: int, T: int, dataset: ShakespeareDataset):
        self.B = B
        self.T = T
        self.dataset = dataset
        self.tokens = torch.tensor(dataset.tokens)
        self.current_position = 0
        
        print(f"DataLoader initialized:")
        print(f"- Batch size: {B}")
        print(f"- Sequence length: {T}")
        print(f"- Total batches per epoch: {len(self.tokens) // (B * T)}")

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]

        if buf.size(0) < B * T + 1:
            self.current_position = 0
            buf = self.tokens[self.current_position : self.current_position + B * T + 1]

        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T

        return x, y

def train_model():
    """Set up and train the model."""
    try:
        if DEVICE.type == 'cuda':
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            torch.set_float32_matmul_precision('high')
        
        torch.manual_seed(1337)
        if DEVICE.type == 'cuda':
            torch.cuda.manual_seed(1337)
        
        total_batch_size = 524288  # ~0.5M tokens
        B = 16  # micro batch size
        T = 1024  # sequence length
        
        print("Initializing training...")
        dataset = ShakespeareDataset()
        train_loader = DataLoaderLite(B=B, T=T, dataset=dataset)
        
        model = GPT(GPTConfig(vocab_size=50304))
        model.to(DEVICE)
        
        if hasattr(torch, 'compile'):
            print("Using torch.compile()")
            model = torch.compile(model)
        
        optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4)
        
        print("Starting training...")
        # Training loop implementation...
        
        return model
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise

def main():
    try:
        print("Welcome to Shakespeare GPT!")
        
        # Initialize bot without requiring command line arguments
        print("\nInitializing chatbot...")
        bot = EnhancedShakespeareBot()
        
        print(bot.get_help_message())
        print("\nReady for your questions!")
        
        while True:
            try:
                query = input("\nYou: ").strip()
                if query.lower() in ['exit', 'quit', 'bye']:
                    print("\nFarewell! Parting is such sweet sorrow...")
                    break
                    
                response = bot.process_query(query)
                print("\nBot:", response)
                
            except KeyboardInterrupt:
                print("\nExiting gracefully...")
                break
            except Exception as e:
                print(f"\nBot: Apologies, I encountered an error: {str(e)}")
                print("Please try again with a different question.")
        
        return 0
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("\nMake sure you have installed all required packages:")
        print("pip install tiktoken torch requests")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
