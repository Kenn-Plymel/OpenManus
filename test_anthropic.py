import os
import anthropic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("ANTHROPIC_API_KEY")

# Verify API key exists
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables. Make sure you've set it in your .env file or environment.")

# Create client with the API key
client = anthropic.Anthropic(api_key=api_key)

# Test each model
models = [
    "claude-3-opus-20240229",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-haiku-20241022"
]

for model in models:
    print(f"\nTesting model: {model}")
    
    try:
        message = client.messages.create(
            model=model,
            max_tokens=1000,
            messages=[
                {"role": "user", "content": f"Hello! You're running as {model}. What can you tell me about yourself?"}
            ]
        )
        
        print("Response:")
        print(message.content[0].text)
        print("\n" + "="*50)
        
    except Exception as e:
        print(f"Error with model {model}: {str(e)}")
        print("\n" + "="*50)
