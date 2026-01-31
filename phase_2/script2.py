# Chains (Multi-Step Reasoning)
# Goal -- 1.Rewrite question simply, 2.Answer rewritten question


from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

# LLM model
llm = ChatOllama(
    model="gemma3:1b",
    temperature=0.3
)

# step 1
rewrit_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
        Rewrite the question in a simple way:\n{question}
"""
)

# step 2
answer_rewritten_q = PromptTemplate(
    input_variables=["simplified_q"],
    template="""
        Answer the following clearly:\n{simplified_q}
"""
)
q = "Can you explain how Python lists work?"
simplified = llm.invoke(
    rewrit_prompt.format(question=q)
).content
print(f"--- rewrite content from step 1:---\n{simplified}")

response = llm.invoke(
    answer_rewritten_q.format(simplified_q=simplified)
).content

print(f"---Final output is:---\n{response}")





