import requests
import json
import time
import logging

# Set up logging
logging.basicConfig(filename='pmll_cross_talk.log', level=logging.INFO)

# Define the API keys
LLAMA_API_KEY = "LA-8c4003a74c5040b2b735866f22e754ed55c2ab712b0346b3bca0f1993362704a"
OPENAI_API_KEY = "sk-proj-
JvOfpMYikesYXuIi32gMuyoyamYwAkx6O3PiDFNwSlIsLZCQ9LEFwu_6vjDiQ6KQ4r6dW_hmSgT3BlbkFJshH4vDAndi1Nh3vuN5fzielvukMjHsHyxaKp1AQQuTMSPeE7pI-FbpFCeeGPIRphVvGWFtKV0A"

# Define the API endpoints
LLAMA_API_URL = "https://api.llama.ai/v1/chat"
OPENAI_API_URL = "https://api.openai.com/v1/completions"

# Define the PMLL logic loop
class PMLLState:
    def __init__(self):
        self.iteration_count = 0
        self.last_value = 1.0
        self.last_input = ""

class PMLL:
    def __init__(self):
        self.state = PMLLState()

    def initialize_state(self):
        self.state.iteration_count = 0
        self.state.last_value = 1.0
        self.state.last_input = ""

    def save_state(self):
       <span class="ml-2" /><span class="inline-block w-3 h-3 rounded-full bg-neutral-a12 align-middle mb-[0.1rem]" />
