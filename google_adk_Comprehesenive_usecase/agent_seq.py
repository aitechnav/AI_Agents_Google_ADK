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


#
#sequential agent
# Initialize the Writer Agent
writer_agent = LlmAgent(
     name="WriterAgent",
     model=model,
     instruction="""You are a Writer Agent. Write a comprehensive article about the given topic and also use information from 'written_results'.""",
     description="Write Comprehensive Article",
     output_key="written_results"
 )

# # Initialize the Summarizer Agent
summarizer_agent = LlmAgent(
     name="SummarizeContent",
     model=model,
     instruction="Ask user if they want summary of the article. Summarize the webpage content found in 'written_results'. Do not summarize if user does not want it.",
     output_key="summary"
 )

# # Initialize the Translator Agent
translator_agent = LlmAgent(
     name="TranslatorAgent",
     model=model,
     instruction="""You are an expert Translator Agent. Convert the given text and document from the 'written results' to the target language. Ask user to the type of language they want to translate to.""",
     description="Translate Comprehensive Article",
 )


root_agent = SequentialAgent(
     name = "ResearcherWriterPipelineAgent",
     sub_agents = [writer_agent, summarizer_agent, translator_agent])
#    The agent will run in the defined order provided : Writer -> Sumarizer/Translator)