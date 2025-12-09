FROM python:3.12.12-trixie
WORKDIR /app
COPY projet_rail_estate ./projet_rail_estate 
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
CMD uvicorn projet_rail_estate.api.fast:app --host 0.0.0.0 --port $PORT