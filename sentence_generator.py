import json
import random

class SentenceGenerator:
    def __init__(self, vocab_file):
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)

    def get_random_word(self):
        return random.choice(list(self.vocab.keys()))

    def generate_sentence(self, length):
        sentence = []
        for _ in range(length):
            word = self.get_random_word()
            sentence.append(word)
        return ' '.join(sentence)

    def generate_string(self, length):
        string = []
        for _ in range(length):
            word = self.get_random_word()
            string.append(word)
        return ''.join(string)

def main():
    vocab_file = 'vocab.json'
    generator = SentenceGenerator(vocab_file)

    while True:
        print("1. Generate sentence")
        print("2. Generate string")
        print("3. Quit")
        choice = input("Enter your choice: ")

        if choice == '1':
            length = int(input("Enter the length of the sentence: "))
            sentence = generator.generate_sentence(length)
            print(sentence)
        elif choice == '2':
            length = int(input("Enter the length of the string: "))
            string = generator.generate_string(length)
            print(string)
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
