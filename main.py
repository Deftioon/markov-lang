from embeddings import *

# Create word embeddings
model = create_word_embeddings()

# control experiment
good = Word("good", model)
great = Word("great", model)
bad = Word("bad", model)
terrible = Word("terrible", model)

similarity = good.calculate_cosine_similarity(great)
print(f"Similarity between good and great: {similarity}")

similarity = bad.calculate_cosine_similarity(terrible)
print(f"Similarity between bad and terrible: {similarity}")

similarity = good.calculate_cosine_similarity(bad)
print(f"Similarity between good and bad: {similarity}")

similarity = great.calculate_cosine_similarity(terrible)
print(f"Similarity between great and terrible: {similarity}")


markov_sentence = Sentence("a drug that sends the series time machine with noted similarities", model)
gpt_sentence = Sentence("drug induced time travel allows humans to inhabit the consciousness of relatives, initially the modus operandi for the Trancers series", model)

similarity = markov_sentence.calculate_cosine_similarity(gpt_sentence)
print(similarity)