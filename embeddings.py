import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

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
    
    def calculate_cosine_vectors(self, vector):
        dot_product = np.dot(self.vector, vector)
        magnitude1 = np.linalg.norm(self.vector)
        magnitude2 = np.linalg.norm(vector)
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        cosine_similarity = dot_product / (magnitude1 * magnitude2)
        return cosine_similarity
    
    def calculate_euclidean_distance(self, word):
        return np.linalg.norm(self.vector - word.vector)
    
    def calculate_angle(self, word):
        dot_product = np.dot(self.vector, word.vector)
        magnitude1 = np.linalg.norm(self.vector)
        magnitude2 = np.linalg.norm(word.vector)
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        cosine_similarity = dot_product / (magnitude1 * magnitude2)
        angle_radians = np.arccos(cosine_similarity)
        angle_degrees = np.degrees(angle_radians)
        return angle_degrees
    
    def __add__(self, word):
        return Vector(self.vector + word.vector)
    
    def __sub__(self, word):
        return Vector(self.vector - word.vector)

class Word(Vector):
    def __init__(self, word, model):
        super().__init__(model.encode(word))
        self.word = word

    def __repr__(self) -> str:
        return f"{self.word}: {self.vector}"

class Sentence(Vector):
    def __init__(self, text, model):
        self.text = text
        self.words = text.split()
        self.vectors = [Word(word, model) for word in self.words]
        super().__init__(model.encode(text))

    def __repr__(self) -> str:
        return f"{self.text}: {self.vectors}"
    
def create_word_embeddings():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

def plot_vectors(vectors, labels, title, colors):
    numpy_vectors = [vector.as_numpy() for vector in vectors]
    plt.figure(figsize=(10, 10))
    for i, vector in enumerate(numpy_vectors):
        plt.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color=colors[i], label=labels[i])
        
    plt.xlim(-0.1, 0.1)
    plt.ylim(-0.025, 0.1)
    plt.axhline(0, color='grey', lw=0.5)
    plt.axvline(0, color='grey', lw=0.5)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.legend()
    plt.title(title)
    plt.show()

# Example usage
if __name__ == "__main__":
    model = create_word_embeddings()

    hitler = Word("Hitler", model)
    print(hitler)

    sk = Word("South Korea", model)
    korean = Word("Korean", model)
    japanese = Word("Japanese", model)
    jp = Word("Japan", model)

    result = sk - korean + japanese

    print(jp.calculate_cosine_similarity(result))
    print(jp.calculate_cosine_similarity(japanese))

    # Plot the vectors
    plot_vectors([sk, korean, japanese, jp, result], ["South Korea", "Korean", "Japanese", "Japan", "South Korea - Korean + Japanese"], "Word Embeddings", ["red", "blue", "green", "purple", "orange"])