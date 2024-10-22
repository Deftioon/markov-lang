import gensim
import numpy as np
from nltk.corpus import brown

class Vector:
    def __init__(self, data):
        self.vector = data
    
    def __repr__(self) -> str:
        return f"{self.vector}"
    
    def as_numpy(self):
        return np.array(self.vector)
    
    def calculate_cosine_similarity(self, word):
        dot_product = np.dot(self.vector, word.vector)
        magnitude1 = np.linalg.norm(self.vector)
        magnitude2 = np.linalg.norm(word.vector)
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        cosine_similarity = dot_product / (magnitude1 * magnitude2)
        return cosine_similarity

class Word(Vector):
    def __init__(self, word, model):
        if word in model.wv:
            super().__init__(model.wv[word])
        else:
            super().__init__(np.zeros(model.vector_size))
        self.word = word

    def __repr__(self) -> str:
        return f"{self.word}: {self.vector}"

class Sentence(Vector):
    def __init__(self, text, model):
        self.text = text
        self.words = text.split()
        self.vectors = [Word(word, model) for word in self.words]

        super().__init__(self.calculate_sentence_vector())

    def __repr__(self) -> str:
        return f"{self.text}: {self.vectors}"
    
    def calculate_sentence_vector(self):
        n = len(self.words)
        sentence_vector = np.zeros_like(self.vectors[0].as_numpy())
        for i, word_vector in enumerate(self.vectors):
            freq = self.words.count(word_vector.word)
            sentence_vector += (1 / freq) * word_vector.as_numpy()
        sentence_vector /= n
        return sentence_vector
    
def create_word_embeddings():
    corpus = brown.sents()
    model = gensim.models.Word2Vec(corpus, min_count=1)
    return model

# Example usage
model = create_word_embeddings()

hello = Word("horrible", model)
hi = Word("bad", model)
sentence1 = Sentence("that was a bad movie it was horrible", model)
sentence2 = Sentence("the movie was so good i love it", model)

cosine_similarity = hello.calculate_cosine_similarity(hi)

print(f"The cosine similarity between '{hello.word}' and '{hi.word}' is: {cosine_similarity}")

cosine_similarity = sentence1.calculate_cosine_similarity(sentence2)
print(f"The cosine similarity between '{sentence1.text}' and '{sentence2.text}' is: {cosine_similarity}")