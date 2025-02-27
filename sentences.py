from sentence_transformers import SentenceTransformer

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# List of sentences to be converted into embeddings
sentences = [
    "This is an example sentence.",
    "Sentence transformers are very useful.",
    "Let's convert these sentences into embeddings."
]

# Convert sentences to embeddings
embeddings = model.encode(sentences)

# Print the embeddings
for sentence, embedding in zip(sentences, embeddings):
    print(f"Sentence: {sentence}")
    print(f"Embedding: {embedding}\n")
    print(type(embedding))