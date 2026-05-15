FROM python:3.12-slim

WORKDIR /app

COPY gateway/pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY gateway/lna_agent_gateway lna_agent_gateway/

EXPOSE 9010

CMD ["python3", "-m", "lna_agent_gateway.app"]
