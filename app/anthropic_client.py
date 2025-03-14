from typing import Dict, List, Optional, Union, Any
import json
import asyncio

from anthropic import AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_random_exponential

from app.config import LLMSettings, config
from app.logger import logger
from app.schema import Message, ROLE_VALUES


class AnthropicClient:
    """Client for interacting with Anthropic's Claude API."""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        """
        Initialize Anthropic client.
        
        Args:
            api_key: Anthropic API key
            base_url: Optional base URL for the API (default: None)
        """
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        
        self.client = AsyncAnthropic(**client_kwargs)
    
    @staticmethod
    def _convert_openai_messages_to_anthropic(messages: List[dict]) -> List[dict]:
        """
        Convert OpenAI-style messages to Anthropic format.
        
        Args:
            messages: List of messages in OpenAI format
            
        Returns:
            List of messages in Anthropic format
        """
        converted = []
        system_content = None
        
        # First pass to extract system message
        for msg in messages:
            if msg["role"] == "system":
                if system_content is None:
                    system_content = msg["content"]
                else:
                    # Concatenate multiple system messages
                    system_content += "\n\n" + msg["content"]
        
        # Convert other messages
        for msg in messages:
            if msg["role"] == "system":
                continue  # Skip system messages in this pass
            
            if msg["role"] == "user":
                converted.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                converted.append({"role": "assistant", "content": msg["content"]})
            elif msg["role"] == "tool":
                # Handle tool messages - convert to a content message from the assistant
                # This is an approximation, as Anthropic doesn't have direct tool response handling
                tool_content = f"Tool response: {msg.get('content', 'No content')}"
                if "tool_call_id" in msg:
                    tool_content = f"Tool ID {msg['tool_call_id']}: {tool_content}"
                
                # Add as user message since tool responses are inputs to the assistant
                converted.append({"role": "user", "content": tool_content})
        
        return converted, system_content
    
    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    async def create_message(
        self,
        messages: List[dict],
        max_tokens: int = 4096,

       # Add this right before the API call in create_message method
        print(f"DEBUG: Using Anthropic API key: {'*' * (len(self.client.api_key) - 4) + self.client.api_key[-4:]}")
        print(f"DEBUG: Base URL: {self.client.base_url}")
        print(f"DEBUG: Using model: {params['model']}") 

        
        temperature: float = 0.0,
        stream: bool = False,
        system_prompt: Optional[str] = None,
        tools: Optional[List[dict]] = None,
        **kwargs
    ) -> Union[str, dict]:
        """
        Create a message using Anthropic's API.
        
        Args:
            messages: List of messages in OpenAI format
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            system_prompt: System prompt to use
            tools: Optional tools to use (not fully supported in this implementation)
            **kwargs: Additional parameters for the API
            
        Returns:
            Response from Anthropic API
        """
        try:
            # Convert messages to Anthropic format
            anthropic_messages, extracted_system = self._convert_openai_messages_to_anthropic(messages)
            
            # Use provided system prompt or extracted one
            final_system = system_prompt or extracted_system or "You are a helpful AI assistant."
            
            # Log warning if tools are provided but not using them
            if tools:
                logger.warning("Tools are not fully supported with Anthropic API in this implementation")
                # We could potentially add tool descriptions to the system prompt as a workaround
                tool_descriptions = "\n\n".join([
                    f"Tool: {tool['function']['name']}\nDescription: {tool['function']['description']}"
                    for tool in tools if "function" in tool
                ])
                if tool_descriptions:
                    final_system += f"\n\nYou have access to the following tools:\n{tool_descriptions}"
                    
                    # Add instruction for tool usage format
                    final_system += "\n\nWhen you need to use a tool, respond in this format exactly:" 
                    final_system += '\n<tool_call>\n{"name": "tool_name", "parameters": {"param1": "value1"}}\n</tool_call>'
            
            # Create message params
            params = {
                "model": kwargs.get("model", "claude-3-7-sonnet-20250201"),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": final_system,
                "messages": anthropic_messages,
                "stream": stream,
            }
            
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}
            
            if stream:
                # Streaming implementation
                response_stream = await self.client.messages.create(**params)
                collected_content = []
                
                async for chunk in response_stream:
                    if chunk.type == "content_block_delta":
                        chunk_text = chunk.delta.text
                        collected_content.append(chunk_text)
                        print(chunk_text, end="", flush=True)
                
                print()  # Newline after streaming
                return "".join(collected_content)
            else:
                # Non-streaming implementation
                response = await self.client.messages.create(**params)
                if response.content:
                    return response.content[0].text
                else:
                    raise ValueError("Empty response from Anthropic API")
        
        except Exception as e:
            logger.error(f"Error in Anthropic API call: {str(e)}")
            raise
