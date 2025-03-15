import os
from openai import OpenAI
import anthropic

class ModelHandler:
    def __init__(self, app):
        self.app = app
        # Initialize API clients
        self.openai_client = None
        self.anthropic_client = None
        
        # Only initialize clients if their respective keys are available
        if app.config.get('OPENAI_API_KEY'):
            try:
                self.openai_client = OpenAI(api_key=app.config.get('OPENAI_API_KEY'))
                app.logger.info("OpenAI client initialized successfully")
            except Exception as e:
                app.logger.error(f"Failed to initialize OpenAI client: {e}")
        else:
            app.logger.warning("OpenAI API key not found. OpenAI features will be unavailable.")
        
        if app.config.get('ANTHROPIC_API_KEY'):
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=app.config.get('ANTHROPIC_API_KEY'))
                app.logger.info("Anthropic client initialized successfully")
            except Exception as e:
                app.logger.error(f"Failed to initialize Anthropic client: {e}")
        else:
            app.logger.warning("Anthropic API key not found. Anthropic features will be unavailable.")
    
    def process_request(self, model_family, model_name, user_message, conversation_history=None):
        if model_family == 'openai':
            # Check if OpenAI client is available before processing
            if not self.openai_client:
                raise ValueError("OpenAI client not initialized. API key may be missing or invalid.")
            return self._process_openai_request(model_name, user_message, conversation_history)
        elif model_family == 'anthropic':
            # Check if Anthropic client is available before processing
            if not self.anthropic_client:
                raise ValueError("Anthropic client not initialized. API key may be missing or invalid.")
            return self._process_anthropic_request(model_name, user_message, conversation_history)
        else:
            raise ValueError(f"Unsupported model family: {model_family}")
    
    def _process_openai_request(self, model_name, user_message, conversation_history=None):
        # No need to check again, as we've already verified in process_request
        messages = []
        if conversation_history:
            for exchange in conversation_history:
                messages.append({"role": "user", "content": exchange['user_message']})
                if 'assistant_message' in exchange:
                    messages.append({"role": "assistant", "content": exchange['assistant_message']})
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            self.app.logger.info(f"Using OpenAI model: {model_name}")
            response = self.openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            self.app.logger.error(f"OpenAI API Error: {str(e)}")
            raise
    
    def _process_anthropic_request(self, model_name, user_message, conversation_history=None):
        # No need to check again, as we've already verified in process_request
        # Build messages in Anthropic format
        messages = []
        if conversation_history:
            for exchange in conversation_history:
                messages.append({"role": "user", "content": exchange['user_message']})
                if 'assistant_message' in exchange:
                    messages.append({"role": "assistant", "content": exchange['assistant_message']})
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            self.app.logger.info(f"Using Anthropic model: {model_name}")
            response = self.anthropic_client.messages.create(
                model=model_name,
                max_tokens=4000,
                messages=messages,
                temperature=0.7
            )
            return response.content[0].text
        except Exception as e:
            self.app.logger.error(f"Anthropic API Error: {str(e)}")
            raise
