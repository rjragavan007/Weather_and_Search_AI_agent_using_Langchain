import os
import streamlit as st
from dotenv import load_dotenv
import certifi
from langchain_groq import ChatGroq
from langchain import hub
from langchain.tools import tool
import requests
from langchain.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------- Setup (runs once, cached) ----------
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
WEATHERSTACK_API_KEY = os.environ["WEATHERSTACK_API"]


@tool
def weather_tool(city: str) -> str:
    """Fetch current weather information"""
    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )
    response = requests.get(url=url)
    data = response.json()

    if "current" not in data:
        return "Data not found"

    return (
        f"city: {city} "
        f"Weather: {data['current']['temperature']} celsius "
        f"Humidity: {data['current']['humidity']} %"
    )


@st.cache_resource(show_spinner=False)
def build_agent():
    search_tool = TavilySearchResults(max_results=2)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0.5, google_api_key=GEMINI_API_KEY
    )
    prompt = hub.pull("hwchase17/react")
    tools = [search_tool, weather_tool]
    agent = create_react_agent(llm=llm, prompt=prompt, tools=tools)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)


# ---------- Page config ----------
st.set_page_config(page_title="Research & Weather Agent", page_icon="🌦️", layout="centered")

st.title("🌦️ Research & Weather Agent")
st.caption("Ask a question that needs a web search, a weather lookup, or both — the agent decides which tools to use.")

with st.sidebar:
    st.header("About")
    st.write(
        "This agent uses a **ReAct** pattern (Reason + Act) with two tools:\n\n"
        "- 🔎 **Tavily Search** — for general/current info\n"
        "- ☁️ **Weather Tool** — live weather via Weatherstack"
    )
    st.divider()
    example = st.selectbox(
        "Try an example",
        [
            "",
            "Find the capital of America and find its weather",
            "What's the weather in Tokyo right now?",
            "Who won the last F1 race and what's the weather there?",
        ],
    )
    show_steps = st.toggle("Show agent's reasoning steps", value=False)

# ---------- Chat-style history ----------
if "history" not in st.session_state:
    st.session_state.history = []

for entry in st.session_state.history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        if entry["role"] == "assistant" and entry.get("steps") and show_steps:
            with st.expander("🧠 Reasoning steps"):
                for i, (action, observation) in enumerate(entry["steps"], 1):
                    st.markdown(f"**Step {i}**")
                    st.code(f"Tool: {action.tool}\nInput: {action.tool_input}", language="text")
                    st.markdown(f"*Observation:* {observation}")

# ---------- Input ----------
query = st.chat_input("Ask something...") or (example if example else None)

if query:
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking and using tools..."):
            try:
                agent_exc = build_agent()
                response = agent_exc.invoke({"input": query})
                answer = response["output"]
                steps = response.get("intermediate_steps", [])
            except Exception as e:
                answer = f"⚠️ Something went wrong: {e}"
                steps = []

        st.markdown(answer)
        if steps and show_steps:
            with st.expander("🧠 Reasoning steps"):
                for i, (action, observation) in enumerate(steps, 1):
                    st.markdown(f"**Step {i}**")
                    st.code(f"Tool: {action.tool}\nInput: {action.tool_input}", language="text")
                    st.markdown(f"*Observation:* {observation}")

    st.session_state.history.append({"role": "assistant", "content": answer, "steps": steps})