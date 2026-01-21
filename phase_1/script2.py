# prompt template exercise
from langchain_core.prompts import PromptTemplate
import logging

logging.basicConfig(
    filename="phase_1/app.log",
    level=logging.INFO,
    format="[%(asctime)s -- %(levelname)s]--> %(message)s"
)

promts = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple words within three sentences for beginners. Make three bullet points for each answer."
)
# print(promts)

final_p = promts.format(topic="C++")
# print(final_p)
logging.info(final_p)

from langchain_ollama import ChatOllama

llm_chat = ChatOllama(
    model="gemma3:1b",
    temperature=0.2
)

response = llm_chat.invoke(final_p)

try:
    logging.info(response.content)
except Exception as e:
    logging.error(e)


