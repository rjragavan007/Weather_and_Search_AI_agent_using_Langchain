import os
from dotenv import load_dotenv
import certifi
from langchain_groq import ChatGroq
from langchain import hub
from langchain.tools import tool
import requests
from langchain.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor,create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["SSL_CERT_FILE"]=certifi.where()
load_dotenv()
GEMINI_API_KEY=os.environ["GEMINI_API_KEY"]
TAVILY_API_KEY=os.environ["TAVILY_API_KEY"]
WEATHERSTACK_API_KEY=os.environ["WEATHERSTACK_API"]

# tool1
search_tool=TavilySearchResults(max_results=2)

# tool2
@tool
def weather_tool(city:str)->str:
    """Fetch current weather information"""
    url = (
    f"https://api.weatherstack.com/current?"
    f"access_key={WEATHERSTACK_API_KEY}&query={city}")

    response=requests.get(url=url)
    data=response.json()

    if "current" not in data:
        return "Data not found"

    return (f"city: {city}"
            f"Weather: {data['current']['temperature']} celsius"
            f"Humidity: {data['current']['humidity']} %")

# LLM
llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.5,google_api_key=GEMINI_API_KEY)

# prompt
prompt=hub.pull("hwchase17/react")

# tools collection
tools=[search_tool,weather_tool]

#create agent 
agent=create_react_agent(llm=llm,prompt=prompt,tools=tools)

# agent_execution
agent_exc=AgentExecutor(agent=agent,tools=tools,verbose=True)

response=agent_exc.invoke({"input":("find capital of America and find its weather")})
print(response['output'])