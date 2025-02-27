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
print("Pandas imported.")

# Import Dataset

print("Importing dataset...")

with open('paragraphs.txt', 'r') as file:
    paragraphs = file.readlines()
    paragraphs = [paragraph.lower() for paragraph in paragraphs]

df = pd.DataFrame(paragraphs, columns=['paragraph'])

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

def retrieve_gpt():
    gpt_model = GPT()
    return gpt_model

def process(paragraph, model):
    markov_chain = train_markov_chain(paragraph)
    gpt_model = retrieve_gpt()

    seed = paragraph.split()[0]

    markov_text = markov_chain.generate_text_sequential(seed=seed, num_words=100)
    gpt_text = gpt_model.predict(paragraph, seed=seed)

    original = Sentence(paragraph, model)
    markov_sentence = Sentence(markov_text, model)
    gpt_sentence = Sentence(gpt_text, model)

    original_markov = original.calculate_cosine_similarity(markov_sentence)
    original_gpt = original.calculate_cosine_similarity(gpt_sentence)
    markov_gpt = markov_sentence.calculate_cosine_similarity(gpt_sentence)

    similarities = np.array([original_gpt, original_markov, markov_gpt])

    return similarities

print("Models defined.")

# Process Dataset

print("Processing dataset...")

tracker = pd.DataFrame(columns=['time', 'paragraph_num', 'original_gpt', 'average_gpt_similarity', 'original_markov', 'average_markov_similarity', 'markov_gpt', 'average_markov_gpt_similarity', 'iteration_time'])

original_gpt_sum = 0
original_markov_sum = 0
markov_gpt_sum = 0

i = 0
num_files = 0

random_paragraph = df['paragraph'][2343]
markov_chain = train_markov_chain(random_paragraph)
np.save('markov_chain.npy', markov_chain.transition_matrix_normalized)

while True:
    try:
        start_time = time.time()
        
        paragraph = df['paragraph'][i]
        similarities = process(paragraph, embedding_model)

        original_gpt_sum += similarities[0]
        original_markov_sum += similarities[1]
        markov_gpt_sum += similarities[2]

        new_row = pd.DataFrame([{
            'time': time.time(),
            'paragraph_num': i,
            'original_gpt': similarities[0],
            'average_gpt_similarity': original_gpt_sum / (i + 1),
            'original_markov': similarities[1],
            'average_markov_similarity': original_markov_sum / (i + 1),
            'markov_gpt': similarities[2],
            'average_markov_gpt_similarity': markov_gpt_sum / (i + 1),
            'iteration_time': time.time() - start_time
        }])
        tracker = pd.concat([tracker, new_row], ignore_index=True)

        print(f"Paragraph {i} \t Original GPT: {similarities[0]:.3f} \t Average GPT: {original_gpt_sum / (i + 1):.3f} \t Original Markov: {similarities[1]:.3f} \t Average Markov: {original_markov_sum / (i + 1):.3f} \t Markov GPT: {similarities[2]:.3f} \t Average Markov GPT: {markov_gpt_sum / (i + 1):.3f} \t Iteration Time: {time.time() - start_time:.2f} seconds")
        i += 1

        if i % 100 == 0:
            raise Exception("Once every hundred paragraph backup")

    except Exception as e:
        tracker.to_csv(f'data/tracker{num_files}.csv', index=False)
        num_files += 1
        print(f"Error processing paragraph {i}: {e}")
        print("Retrying...")

 