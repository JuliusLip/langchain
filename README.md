# langchain SQL agent

## Steps for initializing the app locally:
1. Create a virtual environment: `python3 -m venv .venv`
2. Activate virtual environment: `source .venv/bin/activate`
3. Install all the required dependencies into your venv: `pip install -r requirements.txt`
4. Create `.env` file and populate your environment variables there:
	`OPENAI_API_KEY="<your_api_key>"`
5. Start Streamlit app by running: `streamlit run sql_agent.py`