# PromptTemplate
import logging
from langchain_core.prompts import PromptTemplate

logging.basicConfig(
    filename="phase_1/app.log",
    level=logging.INFO,
    format="[%(asctime)s -- %(levelname)s]--> %(message)s"
)

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple words for begineer"
)

final_prompt = prompt.format(topic="machine learning")
# print(final_prompt)
logging.info(final_prompt)

# creating llm
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="gemma3:1b",
    temperature=0.2
)

response = llm.invoke(final_prompt)
logging.info(response.content)


# This is how real apps work.




