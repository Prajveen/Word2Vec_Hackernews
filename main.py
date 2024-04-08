import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import random

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_word):
        embedded = self.embeddings(context_word)
        output = self.linear(embedded)
        return output

class Word2VecTrainer:
    def __init__(self, corpus, vocab_size, window_size, embedding_dim, learning_rate):
        self.corpus = corpus
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.model = Word2Vec(vocab_size, embedding_dim)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

    def generate_training_data(self):
        training_data = []
        for i, target_word in enumerate(self.corpus):
            start = max(0, i - self.window_size)
            end = min(len(self.corpus), i + self.window_size + 1)
            context_words = [self.corpus[j] for j in range(start, end) if j != i]
            for context_word in context_words:
                training_data.append((target_word, context_word))
        return training_data

    def train(self, num_epochs):
        training_data = self.generate_training_data()
        for epoch in range(num_epochs):
            total_loss = 0
            random.shuffle(training_data)
            for target_word, context_word in training_data:
                self.optimizer.zero_grad()
                target_tensor = torch.LongTensor([self.word_to_index[target_word]])
                context_tensor = torch.LongTensor([self.word_to_index[context_word]])
                output = self.model(context_tensor)
                loss = self.loss_function(output, target_tensor)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss}')

    def get_word_embeddings(self):
        return self.model.embeddings.weight.detach().numpy()

# Example usage:
corpus = ['king', 'queen', 'man', 'woman', 'king', 'queen']
word_to_index = {word: idx for idx, word in enumerate(set(corpus))}
vocab_size = len(word_to_index)
window_size = 2
embedding_dim = 100
learning_rate = 0.01

trainer = Word2VecTrainer(corpus, vocab_size, window_size, embedding_dim, learning_rate)
trainer.train(num_epochs=100)
word_embeddings = trainer.get_word_embeddings()
print(word_embeddings)
