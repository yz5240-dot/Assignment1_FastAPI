
import random


class BigramModel:
    def __init__(self, corpus):
        self.bigrams = {}

        for sentence in corpus:
            words = sentence.lower().split()

            for i in range(len(words) - 1):
                current_word = words[i]
                next_word = words[i + 1]

                if current_word not in self.bigrams:
                    self.bigrams[current_word] = []

                self.bigrams[current_word].append(next_word)

    def generate_text(self, start_word, length):
        current_word = start_word.lower()
        generated_words = [current_word]

        for _ in range(length - 1):
            if current_word not in self.bigrams:
                break

            next_word = random.choice(self.bigrams[current_word])
            generated_words.append(next_word)
            current_word = next_word

        return " ".join(generated_words)
