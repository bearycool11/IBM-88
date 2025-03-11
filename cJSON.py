import ctypes
import json

# Load the shared library
lib = ctypes.CDLL('./json.so')

# Define the function prototypes
lib.json_parse.argtypes = [ctypes.c_char_p]
lib.json_parse.restype = ctypes.c_void_p

lib.json_stringify.argtypes = [ctypes.c_void_p]
lib.json_stringify.restype = ctypes.c_char_p

lib.json_get_string.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.json_get_string.restype = ctypes.c_char_p

# Read the contents of vocab.json into a string
with open('vocab.json', 'r') as file:
    json_str = file.read()

# Parse the JSON string into a cJSON object
json_obj = lib.json_parse(json_str.encode())

# Convert the cJSON object back into a JSON string
json_str2 = lib.json_stringify(json_obj).decode()

# Parse the JSON string into a Python dictionary
dict = json.loads(json_str2)

# Retrieve the value of a string by key
value = dict.get('key')

print(value)
