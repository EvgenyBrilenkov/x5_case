FROM postgres:16

# downdload pg-vector
RUN apt update -y && \
    apt install -y postgresql-16-pgvector && \
    echo "installed pgvector" || echo "not installed pgvector"
