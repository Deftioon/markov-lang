print("Importing time...")
import time
print("Time imported.")

print("Importing embeddings...")
from embeddings import *
print("Embeddings imported.")

print("Importing gpt...")
from gpt import *
print("GPT imported.")

print("Importing markov...")
from markov import *
print("Markov imported.")

print("Importing pandas...")
import pandas as pd
import re
import numpy as np
print("Pandas imported.")

# Import Dataset

print("Importing dataset...")

with open('paragraphs.txt', 'r') as file:
    paragraphs = file.readlines()

    # Concatenate paragraphs into a single string
    huge_text = ''.join(paragraphs)
    # Remove escape characters
    huge_text = re.sub(r'\s+', ' ', huge_text)

    # Cut huge_text in half
    half_length = len(huge_text) // 100
    huge_text = huge_text[:half_length]

print("Dataset imported.")

# Create word embeddings

print("Creating word embeddings...")

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Word embeddings created.")

# Model Definitions

print("Defining models...")

def train_markov_chain(paragraph):
    markov_chain = markov(paragraph)
    markov_chain.fit()
    return markov_chain

markov_chain = train_markov_chain(huge_text)

np.save('model/markov_chain_transitions.npy', markov_chain.transitions)
np.save('model/markov_chain_transition_matrix.npy', markov_chain.transition_matrix)
np.save('model/markov_chain_unique_words.npy', markov_chain.unique_words)
np.save('model/markov_chain_words.npy', markov_chain.words)
np.save('model/markov_chain_num_words.npy', markov_chain.num_words)
np.save('model/markov_chain_transition_matrix_normalized.npy', markov_chain.transition_matrix_normalized)

print("Markov chain saved.")

#Load Markov Chain

# print("Loading Markov Chain...")

# markov_chain = markov(huge_text)
# markov_chain.transition_matrix_normalized = np.load('model/markov_chain_transition_matrix_normalized.npy')
# markov_chain.unique_words = np.load('model/markov_chain_unique_words.npy').tolist()
# markov_chain.num_words = np.load('model/markov_chain_num_words.npy').tolist()
# markov_chain.words = np.load('model/markov_chain_words.npy').tolist()

# print("Markov Chain loaded.")

# def retrieve_gpt():
#     gpt_model = GPT()
#     return gpt_model    

# def process(paragraph, model, markov_chain):
#     gpt_model = retrieve_gpt()

#     seed = paragraph.split()[0].lower()

#     markov_seq_text = markov_chain.generate_text_sequential(seed=seed, num_words=100)
#     #markov_rec_text = markov_chain.generate_text_recursive(seed=seed, num_words=100)
#     gpt_text = gpt_model.predict(paragraph, seed=seed)

#     original = Sentence(paragraph, model)
#     markov_seq_sentence = Sentence(markov_seq_text, model)
#     #markov_rec_sentence = Sentence(markov_rec_text, model)
#     gpt_sentence = Sentence(gpt_text, model)
    
#     markov_seq_similarity = original.calculate_cosine_similarity(markov_seq_sentence)
#     #markov_rec_similarity = original.calculate_cosine_similarity(markov_rec_sentence)
#     markov_similarity = markov_seq_similarity

#     gpt_similarity = original.calculate_cosine_similarity(gpt_sentence)

#     markov_gpt_seq = markov_seq_sentence.calculate_cosine_similarity(gpt_sentence)
#     #markov_gpt_rec = markov_rec_sentence.calculate_cosine_similarity(gpt_sentence)
#     markov_gpt_similarity = markov_gpt_seq

#     similarities = np.array([gpt_similarity, markov_similarity, markov_gpt_similarity])

#     return similarities

# print("Models defined.")

# # Process Dataset

# print("Processing dataset...")

# tracker = pd.DataFrame(columns=['time', 'paragraph_num', 'markov_similarity', 'gpt_similarity', 'markov_gpt_similarity', 'iteration_time'])

# original_markov_sum = 0
# original_gpt_sum = 0
# markov_gpt_sum = 0

# continued = 4700
# i = 4700
# num_files = 0

# while True:
#     try:
#         start_time = time.time()
        
#         paragraph = paragraphs[i]
#         similarities = process(paragraph, embedding_model, markov_chain)

#         original_gpt_sum += similarities[0]
#         original_markov_sum += similarities[1]
#         markov_gpt_sum += similarities[2]

#         new_row = pd.DataFrame([{
#             'time': time.time(),
#             'paragraph_num': i,
#             'original_gpt': similarities[0],
#             'average_gpt_similarity': original_gpt_sum / (i + 1 - continued),
#             'original_markov': similarities[1],
#             'average_markov_similarity': original_markov_sum / (i + 1 - continued),
#             'markov_gpt': similarities[2],
#             'average_markov_gpt_similarity': markov_gpt_sum / (i + 1 - continued),
#             'iteration_time': time.time() - start_time
#         }])
#         tracker = pd.concat([tracker, new_row], ignore_index=True)

#         print(f"Paragraph {i} \t Original GPT: {similarities[0]:.3f} \t Average GPT: {original_gpt_sum / (i + 1):.3f} \t Original Markov: {similarities[1]:.3f} \t Average Markov: {original_markov_sum / (i + 1):.3f} \t Markov GPT: {similarities[2]:.3f} \t Average Markov GPT: {markov_gpt_sum / (i + 1):.3f} \t Iteration Time: {time.time() - start_time:.2f} seconds")
#         i += 1

#         if i % 100 == 0:
#             raise Exception("Once every hundred paragraph backup")

#     except Exception as e:
#         tracker.to_csv(f'big_data/tracker{num_files}.csv', index=False)
#         num_files += 1
#         print(f"Error processing paragraph {i}: {e}")
#         print("Retrying...")