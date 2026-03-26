import openai
import os
from openai import OpenAI
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    logger.error("OPENROUTER_API_KEY not found in environment variables")
    raise ValueError("OPENROUTER_API_KEY environment variable is required")

openai.api_key = api_key
openai.base_url = "https://openrouter.ai/api/v1"

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

def generate_response(prompt, chat_history=None):
    """
    Generate response from LLM using the provided prompt and optional chat history.
    
    Args:
        prompt (str): Complete prompt with system context and user query
        chat_history (list, optional): List of previous messages for context format:
                                       [{"role": "user"|"assistant", "content": str}, ...]
        
    Returns:
        str: Generated response from LLM
    """
    try:
        messages = [
            {"role": "system", "content": "You are a technical assistant."}
        ]
        
        if chat_history:
            messages.extend(chat_history)
        
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=messages,
            temperature=0.2,
            max_tokens=2000
        )

        return response.choices[0].message.content

    except openai.RateLimitError:
        error_msg = "API rate limit exceeded. Please try again later."
        logger.error(error_msg)
        return error_msg
    
    except openai.AuthenticationError:
        error_msg = "Authentication failed. Check your OPENROUTER_API_KEY."
        logger.error(error_msg)
        return error_msg
    
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        logger.error(error_msg)
        return error_msg