import numpy as np

from core.nn import Module, Embedding, Linear, SelfAttention

class AttentionLanguageModel(Module):
    def __init__(self, vocab_size, context_size, head_size, head_num):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = head_size * head_num
        self.context_size = context_size
        self.head_size = head_size
        self.head_num = head_num

        self.embedding = Embedding(self.vocab_size, self.embedding_dim)
        self.positional_embedding = Embedding(self.context_size, self.embedding_dim)
        self.attention = SelfAttention(self.head_size, self.head_num)
        self.lm_head = Linear(self.embedding_dim, self.vocab_size)

    def __call__(self, tokens, cache=None):
        x = self.embedding(tokens) + self.positional_embedding(np.arange(tokens.shape[1]))
        x = self.attention(x, cache=cache)
        x = x[:, -1, :]
        return self.lm_head(x)
