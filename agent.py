import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import initialize_agent, AgentType


llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)


wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = [wikipedia]


agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)


if __name__ == "__main__":
    query = "Who discovered the structure of DNA and in what year?"
    print(f"User: {query}\n")
    
    # The agent will log its internal Thought/Action/Observation process
    response = agent.run(query)
    print(f"\nFinal Answer: {response}")