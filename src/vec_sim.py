from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

def calculate_cosine_similarity(sentence, word1, word2):
    # Convert the words into vectors
    vector1 = word_to_vector(sentence, word1)
    vector2 = word_to_vector(sentence, word2)

    # Calculate the cosine similarity
    similarity = cosine_similarity(vector1, vector2)

    return similarity

def word_to_vector(word, sentence):
    # Load the Word2Vec model
    model = Word2Vec(sentence, min_count=1)

    # Get the vector representation of the word
    vector = model.wv[word]

    return vector

# Example usage
word1 = "apple"
word2 = "orange"
similarity = calculate_cosine_similarity([["apple"], ["orange"]], word1, word2)
print(f"The cosine similarity between '{word1}' and '{word2}' is: {similarity}")