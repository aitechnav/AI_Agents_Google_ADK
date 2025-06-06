from google.adk.agents import Agent, LlmAgent, SequentialAgent, ParallelAgent
from google.adk.models.lite_llm import LiteLlm
from textblob import TextBlob
import os


# os.environ["OLLAMA_CONTEXT_LENGTH"] = "4096"
os.environ["OLLAMA_KEEP_ALIVE"] = "-1"
os.environ["OPENAI_API_KEY"] = "unused"
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"

USE_OLLAMA = False

OLLAMA_MODEL = "qwen2.5:14b"

model = LiteLlm(model=f"openai/{OLLAMA_MODEL}") if USE_OLLAMA else "gemini-2.0-flash"

#MODEL = LiteLlm(model="groq/llama3-8b-8192")

def get_sentiment(text:str) -> float:
    """
    Get the sentiment of the text using TextBlob.
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity




#root agent
root_agent = Agent(
    name = "root",
    model = model,
    description = """Agent to analyze given text and return the sentiment score. You are a text analysis agent your job is to analyze the given text and return the sentiment score.""",
    tools = [get_sentiment])

