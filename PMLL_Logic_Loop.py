import json
import os
import time
import logging

class PMLLState:
    def __init__(self):
        self.iteration_count = 0
        self.last_value = 1.0
        self.last_input = ""

class PMLL:
    def __init__(self):
        self.state = PMLLState()
        self.state_file = "pmll_state.dat"
        self.log_file = "pmll_log.txt"
        self.logger = logging.getLogger("PMLL")
        self.logger.setLevel(logging.INFO)
        self.handler = logging.FileHandler(self.log_file)
        self.handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(self.handler)

    def initialize_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, "rb") as file:
                self.state = pickle.load(file)
            self.logger.info(f"Resumed state: Iteration {self.state.iteration_count}, Last Value: {self.state.last_value}, Last Input: {self.state.last_input}")
        else:
            self.logger.info("Initialized new state.")

    def save_state(self):
        with open(self.state_file, "wb") as file:
            pickle.dump(self.state, file)
        self.logger.info("State saved successfully.")

    def log_event(self, message):
        self.logger.info(message)

    def novel_topic(self, input_str):
        return input_str != self.state.last_input

    def novel_input(self, input_str):
        if self.novel_topic(input_str):
            self.state.last_input = input_str
            self.logger.info("Detected Novel Input")
            self.save_state()

    def logic_loop(self):
        while True:
            input_str = input("Enter new input (or type 'exit' to quit): ")
            if input_str == "exit":
                self.logger.info("User exited loop.")
                self.save_state()
                break
            self.novel_input(input_str)
            self.state.iteration_count += 1
            self.state.last_value *= 1.1
            log_msg = f"Iteration: {self.state.iteration_count}, Last Value: {self.state.last_value}, Last Input: {self.state.last_input}"
            self.logger.info(log_msg)
            print(log_msg)
            self.save_state()
            time.sleep(1)

if __name__ == "__main__":
    pmll = PMLL()
    pmll.initialize_state()
    pmll.logic_loop()
