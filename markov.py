import numpy as np
import re
import tqdm
import torch

class markov:
    def __init__(self, text):
        self.text = text
        self.words = text.split()
        self.transitions = {}
        self.transition_matrix = None
        self.transition_matrix_normalized = None
        self.unique_words = None
        self.num_words = None
    
    def remove_punctuation(self, text):
        return re.sub(r'[^\w\s]', '', text)
    
    def replace_punctuation(self, text):
        text = re.sub(r'([.,!?])', r' \1 ', text)
        text = re.sub(r',', ' $COMMA$ ', text)
        text = re.sub(r'\.', ' $PERIOD$ ', text)
        text = re.sub(r'!', ' $EXCLAMATION$ ', text)
        text = re.sub(r'\?', ' $QUESTION$ ', text)
        return text
    
    def process_text(self, mode="remove"):
        if mode == "remove":
            text = self.text.lower()
            text = self.remove_punctuation(text)
        elif mode == "replace":
            text = self.text.lower()
            text = self.replace_punctuation(text)
        
        text = re.sub(r'\s+', ' ', text)
        self.words = text.split()

    def count_transitions(self):
        for i in tqdm.tqdm(range(len(self.words) - 1)):
            current_word = self.words[i]
            next_word = self.words[i + 1]

            if current_word not in self.transitions:
                self.transitions[current_word] = {}

            if next_word not in self.transitions[current_word]:
                self.transitions[current_word][next_word] = 0

            self.transitions[current_word][next_word] += 1

    def create_transition_matrix(self):
        self.unique_words = sorted(set(self.words))
        self.num_words = len(self.unique_words)
        self.transition_matrix = np.zeros((self.num_words, self.num_words))

        for i, word in tqdm.tqdm(enumerate(self.unique_words)):
            if word in self.transitions:
                for next_word, count in self.transitions[word].items():
                    j = self.unique_words.index(next_word)
                    self.transition_matrix[i, j] = count

    def normalize_transition_matrix(self):
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=1, keepdims=True)

        self.transition_matrix_normalized = softmax(self.transition_matrix)
        #self.transition_matrix_normalized = np.divide(self.transition_matrix, row_sums[:, np.newaxis], out=np.zeros_like(self.transition_matrix), where=row_sums[:, np.newaxis]!=0)

    def fit(self, mode="remove"):
        self.process_text(mode=mode)
        self.count_transitions()
        self.create_transition_matrix()
        self.normalize_transition_matrix()

    def generate_text_sequential(self, seed=-1, num_words=100):
        if seed == -1:
            seed = np.random.randint(self.num_words)
        #current_word = self.unique_words[seed]
        current_word = seed
        text = current_word

        for i in range(num_words):
            try:
                current_index = self.unique_words.index(current_word)
                next_index = np.random.choice(self.num_words, p=self.transition_matrix_normalized[current_index])
                current_word = self.unique_words[next_index]
                text += ' ' + current_word
            except:
                return current_word

        return text
    
    def generate_text_recursive(self, seed=-1, num_words=100):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if seed == -1:
            seed = np.random.randint(self.num_words)
        text = self.unique_words[seed]
        transition_matrix_power = torch.tensor(self.transition_matrix_normalized).float()
        transition_matrix_power = transition_matrix_power.to(device)

        torch_normalised = torch.tensor(self.transition_matrix_normalized).float()
        torch_normalised = torch_normalised.to(device)
        
        for i in range(num_words):
            transition_matrix_power  = torch.matmul(transition_matrix_power, torch_normalised)
            next_index = torch.multinomial(transition_matrix_power[0], 1)
            #next_index = np.random.choice(self.num_words, p=transition_matrix_power.cpu()[0])
            current_word = self.unique_words[next_index]
            #print(current_word, end=' ')
            text += ' ' + current_word

        return text
    
    def __repr__(self):
        return f" ========================= \n Markov Chain Model trained on text with {self.num_words} unique words and {len(self.words)} total words \n ========================="