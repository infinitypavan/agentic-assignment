import os
import wikipedia
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage


wikipedia.set_user_agent("AgenticAssignmentApp/1.0 (contact@example.com)")

os.environ["GOOGLE_API_KEY"] = "<enter GOOGLE API KEY here"

@tool
def calculator(expression: str) -> str:
    """Useful for evaluating mathematical expressions.
    Input should be a string like '37400000 * 0.15' or '2 ** 10'."""
    try:
        allowed = {"abs": abs, "round": round}
        return str(eval(expression, {"__builtins__": None}, allowed))
    except Exception as e:
        return f"Error evaluating expression: {e}"

wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
web_search = DuckDuckGoSearchRun()

tools = [wikipedia_tool, web_search, calculator]
tools_by_name = {t.name: t for t in tools}

llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)
llm_with_tools = llm.bind_tools(tools)

def run_agent(user_query: str):
    messages = [HumanMessage(content=user_query)]
    
    while True:
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # Stop condition: model returns a final text answer
        if not response.tool_calls:
            return response.content
        
        # Action: Execute tools safely
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"--> Action: Calling '{tool_name}' with args {tool_args}")
            
            tool_obj = tools_by_name[tool_name]
            try:
                observation = tool_obj.invoke(tool_args)
            except Exception as err:
                observation = f"Tool execution failed: {err}. Please try another tool or proceed with existing information."
            
            print(f"<-- Observation: {observation}\n")
            messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))

if __name__ == "__main__":
    query = "What is the population of Tokyo, and what is 15% of that number?"
    print(f"User: {query}\n")
    
    final_answer = run_agent(query)
    print(f"Final Answer:\n{final_answer}")