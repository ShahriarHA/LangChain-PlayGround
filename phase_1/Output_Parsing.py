# this app will say only YES or NO!
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

promptem = PromptTemplate(
    input_variables=["question"],
    template="""
        Return the answer in this format:  <Answer is YES or NO>
        Do not add anything else.

        Question: {question}
        """
)

llm = ChatOllama(
    model="gemma3:1b",
    temperature=0.3
)

q = "Do you sing in the shower?"

# full_prompts = promptem.format(question=q)
# print(full_prompts)

# response = llm.invoke(full_prompts)
# print(response.content)

while True:
    user_input = input("User ---> ")
    if user_input.lower() == "exit":
        break
    else:
        full_prompt = promptem.format(question=user_input)
        response = llm.invoke(full_prompt)
        # print(f"Bot ---> {response.content}")
        try:
            if not response.content.startswith("<Answer is "):
                raise ValueError("invalid format!")
            else:
                print(f"Bot ---> {response.content}")
        except:
            print(f"Your app respond an Invalid Output! output is {response.content}")
        
