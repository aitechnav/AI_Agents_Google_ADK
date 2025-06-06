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



#parallel agent

# Define the keyword extraction agent
keyword_agent = LlmAgent(
    name="KeywordExtractor",
    model=model,
    description="Extract the relevant keywords from the text",
    instruction="Extract main keywords from the provided text.",
    output_key="keywords"
)

# Define the summarization agent
summarizer_agent = LlmAgent(
    name="Summarizer",
    model=model,
    description="Summarize the given text",
    instruction="Create a concise summary of the text.",
    output_key="summary"
)

# Define the sentiment agent
sentiment_agent = LlmAgent(
    name="SentimentAnalyzer",
    model=model,
    description="provide sentiment of  the given text",
    instruction="Provide a sentiment or polarity score of the text.",
    output_key="sentiment"
)

# Define the root agent that runs both agents in parallel
root_agent = ParallelAgent(
    name="ParallelTextAnalysisAgent",
    sub_agents=[sentiment_agent, keyword_agent, summarizer_agent],
    description="Runs multiple specialist agents in parallel to analyse a text"
)
