import src.markov as markov

text = """
Trancers  started out as an homage to pulp detective novels, with noted similarities to other cult sci-fi movies, such as Blade Runner and The Terminator (the latter of which was released the same year). In the series, time travel is initially made possible using a drug that sends the person into the consciousness of a relation, but expanded to include the pre-set co-ordinates of a time machine, with the fourth and fifth films introducing other means of time travel between other dimensions.
"""

m = markov.markov(text)
m.fit(mode="remove")

print(m)

print(m.transition_matrix)

# Generated Sequentially
print(m.generate_text_sequential(seed=0, num_words=10))
