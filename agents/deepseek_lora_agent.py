import os

class DeepSeekLoRAAgent:
    def __init__(self):
        # Replace hardcoded token with environment variable
        self.eas_token = os.getenv("EAS_TOKEN")
        if not self.eas_token:
            raise ValueError("EAS_TOKEN environment variable is not set.")

    def generate_code(self):
        # ... existing code ...