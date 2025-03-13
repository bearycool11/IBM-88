import json
import random
import os

class SentenceGenerator:
    def __init__(self, vocab_file):
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"The file {vocab_file} does not exist.")
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)
        if not isinstance(self.vocab, dict):
            raise ValueError("The vocabulary file must contain a JSON object.")

    def get_random_word(self):
        return random.choice(list(self.vocab.keys()))

    def generate_sentence(self, length):
        if length <= 0:
            raise ValueError("Length must be a positive integer.")
        sentence = []
        for _ in range(length):
            word = self.get_random_word()
            sentence.append(word)
        return ' '.join(sentence)

    def generate_string(self, length):
        if length <= 0:
            raise ValueError("Length must be a positive integer.")
        string = []
        for _ in range(length):
            word = self.get_random_word()
            string.append(word)
        return ''.join(string)

def main():
    vocab_file = 'vocab.json'
    try:
        generator = SentenceGenerator(vocab_file)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return

    while True:
        print("1. Generate sentence")
        print("2. Generate string")
        print("3. Quit")
        choice = input("Enter your choice: ")

        if choice == '1':
            try:
                length = int(input("Enter the length of the sentence: "))
                sentence = generator.generate_sentence(length)
                print(sentence)
            except ValueError as e:
                print(e)
        elif choice == '2':
            try:
                length = int(input("Enter the length of the string: "))
                string = generator.generate_string(length)
                print(string)
            except ValueError as e:
                print(e)
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
