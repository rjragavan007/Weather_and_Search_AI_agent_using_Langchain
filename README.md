conda create -n langagent python =3.11 -y

conda activate langagent

pip install -r requirements.txt

This project is a ReAct agent (built with LangChain and Gemini) that answers questions by intelligently combining live web search and real-time weather lookups, wrapped in an interactive Streamlit interface. Users can ask natural-language questions, and the agent reasons step-by-step, choosing the right tool to fetch accurate, up-to-date answers.
