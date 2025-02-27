import os
from dotenv import load_dotenv
from groq import Groq

class GPT:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GROK_KEY")
        self.client = Groq(api_key=self.api_key)

    def predict(self, prompt, seed):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content":  f"I want the predicted text only, no fillers and nothing else. Predict the text based on '{prompt}' for 100 words and 100 words ONLY beginning with the word '{seed}'. Stay as close to the original meaning of the prompt as you can",
                }
            ],
            model="llama3-8b-8192",
            max_tokens=100,
        )

        return chat_completion.choices[0].message.content

# Example usage
if __name__ == "__main__":
    gpt = GPT()
    prompt = "Trancers  started out as an homage to pulp detective novels, with noted similarities to other cult sci-fi movies, such as Blade Runner and The Terminator (the latter of which was released the same year). In the series, time travel is initially made possible using a drug that sends the person into the consciousness of a relation, but expanded to include the pre-set co-ordinates of a time machine, with the fourth and fifth films introducing other means of time travel between other dimensions."
    seed = "drug"
    predicted_text = gpt.predict(prompt, seed)
    print(predicted_text)