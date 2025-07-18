FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# IMPORTANT: Unset any STREAMLIT environment variables
ENV STREAMLIT_SERVER_PORT=""
ENV STREAMLIT_SERVER_ADDRESS=""

CMD ["python", "run_streamlit.py"]
