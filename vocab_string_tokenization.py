import re

class Tokenizer:
    def __init__(self, input_string):
        self.input_string = input_string
        self.index = 0

    def is_utf8_start_byte(self, byte):
        return (byte & 0xC0) != 0x80

    def is_c_delimiter(self, char):
        delimiters = " \t\n(){}[];,=*+-/<>!&|^~"
        return char in delimiters

    def tokenize(self):
        tokens = []
        while self.index < len(self.input_string):
            # Handle comments (single-line and multi-line)
            if self.input_string[self.index:self.index+2] == "//":
                # Single-line comment
                comment = self.input_string[self.index:]
                tokens.append(comment)
                break
            elif self.input_string[self.index:self.index+2] == "/*":
                # Multi-line comment
                comment = "/*"
                self.index += 2
                while self.index < len(self.input_string) and self.input_string[self.index:self.index+2] != "*/":
                    comment += self.input_string[self.index]
                    self.index += 1
                if self.index < len(self.input_string) and self.input_string[self.index:self.index+2] == "*/":
                    comment += "*/"
                    self.index += 2
                tokens.append(comment)
                continue

            # Handle string literals
            if self.input_string[self.index] == '"':
                string_literal = '"'
                self.index += 1
                while self.index < len(self.input_string) and self.input_string[self.index] != '"':
                    if self.input_string[self.index] == '\\' and self.index + 1 < len(self.input_string):
                        string_literal += self.input_string[self.index:self.index+2]
                        self.index += 2
                    else:
                        string_literal += self.input_string[self.index]
                        self.index += 1
                if self.index < len(self.input_string) and self.input_string[self.index] == '"':
                    string_literal += '"'
                    self.index += 1
                tokens.append(string_literal)
                continue

            # Handle character literals
            if self.input_string[self.index] == '\'':
                char_literal = '\''
                self.index += 1
                while self.index < len(self.input_string) and self.input_string[self.index] != '\'':
                    if self.input_string[self.index] == '\\' and self.index + 1 < len(self.input_string):
                        char_literal += self.input_string[self.index:self.index+2]
                        self.index += 2
                    else:
                        char_literal += self.input_string[self.index]
                        self.index += 1
                if self.index < len(self.input_string) and self.input_string[self.index] == '\'':
                    char_literal += '\''
                    self.index += 1
                tokens.append(char_literal)
                continue

            # Check if the current character is a delimiter
            if self.is_c_delimiter(self.input_string[self.index]):
                # Print the token before the delimiter
                if self.index > 0:
                    token = self.input_string[:self.index]
                    tokens.append(token)
                # Print the delimiter as its own token
                tokens.append(self.input_string[self.index])
                self.index += 1
            else:
                # Handle UTF-8 character traversal or normal characters
                if self.is_utf8_start_byte(ord(self.input_string[self.index])):
                    self.index += 1
                    while self.index < len(self.input_string) and (ord(self.input_string[self.index]) & 0xC0) == 0x80:
                        self.index += 1
                else:
                    self.index += 1
        return tokens

def main():
    input_string = input("Enter a C/C++ code snippet to tokenize: ")
    tokenizer = Tokenizer(input_string)
    tokens = tokenizer.tokenize()
    for token in tokens:
        print(f"Token: {token}")

if __name__ == "__main__":
    main()
