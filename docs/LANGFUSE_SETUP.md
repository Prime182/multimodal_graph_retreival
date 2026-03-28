# LangFuse Tracing Integration Guide

This document describes how to set up and use LangFuse observability and debugging for the GraphRAG project.

## Overview

LangFuse provides comprehensive observability for LLM applications. The GraphRAG project has been instrumented with LangFuse tracing at key points in the pipeline:

- **Entity Extraction (extraction.py)**: Traces LLM calls for entity extraction and logs extraction metrics
- **Query Synthesis (rag.py)**: Traces LLM calls for query synthesis and claim extraction
- **Entity Canonicalization (canonicalization.py)**: Traces entity deduplication and merging
- **Search Operations (search_service.py)**: Traces retrieval operations and search metrics
- **API Endpoints (server.py)**: Traces all HTTP requests to search and synthesize endpoints

## Installation & Setup

### 1. Install LangFuse

LangFuse is included in `requirements.txt`. Ensure it's installed:

```bash
pip install -r requirements.txt
```

Or install directly:

```bash
pip install langfuse>=2.0,<3
```

### 2. Get LangFuse Credentials

1. Go to [Langfuse Cloud](https://cloud.langfuse.com) or self-host LangFuse
2. Create a project
3. Get your API credentials:
   - **Public Key**: Available in project settings
   - **Secret Key**: Available in project settings

### 3. Configure Environment Variables

Set the following environment variables in your `.env` file or shell:

```bash
# LangFuse Configuration
export LANGFUSE_PUBLIC_KEY="pk_xxx..."
export LANGFUSE_SECRET_KEY="sk_xxx..."
export LANGFUSE_HOST="https://cloud.langfuse.com"  # Optional, defaults to cloud
```

For local development with a `.env` file in the project root:

```bash
LANGFUSE_PUBLIC_KEY=pk_xxx...
LANGFUSE_SECRET_KEY=sk_xxx...
LANGFUSE_HOST=https://cloud.langfuse.com
```

Then load it:

```bash
source .venv/bin/activate
export $(cat .env | xargs)
```

## Usage

### Automatic Tracing

Once LangFuse is configured, tracing happens automatically:

1. **Entity Extraction Pipeline**:
   ```python
   from backend.graphrag import extract_layer2
   
   layer2_doc = extract_layer2(paper, use_gemini=True)
   # Traces: individual LLM calls, entity extraction metrics
   ```

2. **Query Synthesis**:
   ```python
   from backend.graphrag import QuerySynthesizer
   
   synthesizer = QuerySynthesizer()
   result = synthesizer.synthesize(question="...", search_results={...})
   # Traces: synthesis LLM call, output metrics
   ```

3. **Search Operations**:
   ```python
   from backend.graphrag import GraphRAGSearchService
   
   search_service = GraphRAGSearchService(input_dir="articles")
   results = search_service.search(query="...", top_k=5)
   # Traces: retrieval operation, result metrics
   ```

4. **API Endpoints**:
   ```bash
   # Start the server
   python -m backend.graphrag.cli serve --port 8000
   
   # Make requests - all are automatically traced
   curl -X POST http://localhost:8000/api/search \
     -H "Content-Type: application/json" \
     -d '{"query": "BERT models", "top_k": 5}'
   ```

### Manual Tracing

For custom operations, use the tracing manager:

```python
from backend.graphrag.tracing import get_tracing_manager

tracer = get_tracing_manager()

# Simple trace
with tracer.trace(name="my_operation", input_data={"param": "value"}) as trace_ctx:
    result = do_something()
    trace_ctx["output"] = result

# Log LLM calls
tracer.log_llm_call(
    name="custom_llm",
    model="models/gemini-2.0-flash",
    prompt="Your prompt here",
    response="Model response",
    tokens_used={"input": 100, "output": 50},
)

# Log retrieval operations
tracer.log_retrieval(
    query="search query",
    results=retrieved_items,
    result_count=len(retrieved_items),
)

# Flush pending traces
tracer.flush()
```

## Tracing Points

### 1. Entity Extraction (`extraction.py`)

**Traced Events:**
- `extract_layer2`: Overall extraction operation
  - Input: paper ID, chunk count
  - Output: entity count, entity types breakdown
- Individual LLM calls to Gemini for each chunk
  - Model: `models/gemini-2.0-flash`
  - Includes: paper ID, chunk ID, section title

**Metrics Logged:**
- Entity count by type (Concept, Method, Claim, Result, Equation)
- Extraction success/failure rates

### 2. Query Synthesis (`rag.py`)

**Traced Events:**
- `query_synthesis`: Synthesis operation
  - Input: question, passage count
  - Output: answer length, confidence score
- `claim_extraction`: Claim extraction from text
  - Input: text length
  - Output: claim count

**Metrics Logged:**
- Synthesis model used
- Input tokens, output tokens
- Confidence scores
- Source attribution

### 3. Entity Canonicalization (`canonicalization.py`)

**Traced Events:**
- `entity_canonicalization`: Overall canonicalization
  - Input: document count
  - Output: mapping size, entity types

**Metrics Logged:**
- Merge count by entity type
- Deduplication effectiveness

### 4. Search Service (`search_service.py`)

**Traced Events:**
- `document_search`: Search operation
  - Input: query, top_k parameter
  - Output: hit counts by type

**Metrics Logged:**
- Text hits, table hits, paper hits
- Entity matches
- Citation matches

### 5. API Endpoints (`server.py`)

**Traced Events:**
- `api_search_get`: GET /api/search
- `api_search_post`: POST /api/search
- `api_synthesize`: POST /api/synthesize
- `api_*_error`: Error events for each endpoint

**Metrics Logged:**
- Query parameters
- Response times
- Error messages
- Result counts

## Viewing Traces

### In LangFuse Dashboard

1. Visit your LangFuse project dashboard
2. Go to "Traces" section
3. Filter by:
   - Trace name (e.g., "query_synthesis", "entity_extraction")
   - Time range
   - Status (success/error)
4. Click on individual traces to see:
   - Full execution timeline
   - Input/output data
   - Token usage (for LLM calls)
   - Metadata and custom fields
   - Error details (if applicable)

### Programmatically

```python
from backend.graphrag.tracing import get_tracing_manager

tracer = get_tracing_manager()

# Check if tracing is enabled
if tracer.enabled:
    print(f"Tracing enabled: {tracer.langfuse}")
else:
    print("LangFuse not configured - set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
```

## Performance Considerations

- **Async Flushing**: LangFuse uses background threads for sending traces (non-blocking)
- **Sampling**: Disable LangFuse by not setting credentials to reduce overhead
- **Custom Host**: For self-hosted LangFuse, set `LANGFUSE_HOST` to your instance

## Troubleshooting

### Traces Not Showing Up

1. **Check Credentials**: Verify `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set:
   ```bash
   echo $LANGFUSE_PUBLIC_KEY
   echo $LANGFUSE_SECRET_KEY
   ```

2. **Check Connection**: Ensure network access to LangFuse host:
   ```bash
   curl https://cloud.langfuse.com/api/health
   ```

3. **Enable Debug Mode**: Run with verbose logging:
   ```bash
   python -c "
   from backend.graphrag.tracing import get_tracing_manager
   tracer = get_tracing_manager()
   print(f'Enabled: {tracer.enabled}')
   print(f'Host: {tracer.langfuse.base_url if tracer.langfuse else None}')
   "
   ```

### High Latency

- LangFuse sends traces in background threads - shouldn't impact app performance
- If slow: reduce call volume or increase flush interval

### Memory Usage

- Traces are batched before sending
- Call `tracer.flush()` periodically if processing large batches

## Integration Examples

### Example 1: Full Pipeline with Tracing

```python
from backend.graphrag import (
    parse_article,
    chunk_article,
    extract_layer2,
    build_layer3,
)
from backend.graphrag.tracing import get_tracing_manager

tracer = get_tracing_manager()

# Parse and process article
article_data = parse_article("path/to/article.xml")
paper = chunk_article(article_data)

with tracer.trace(name="full_pipeline", input_data={"paper_id": paper.paper_id}):
    # Extract entities with tracing
    layer2 = extract_layer2(paper, use_gemini=True)
    
    # Build graph with tracing
    # (tracing added via edges.py instrumentation)
    layer3 = build_layer3([paper], [layer2])

tracer.flush()
```

### Example 2: Custom Search with Tracing

```python
from backend.graphrag import GraphRAGSearchService
from backend.graphrag.tracing import get_tracing_manager

tracer = get_tracing_manager()
search_service = GraphRAGSearchService(input_dir="articles")

queries = [
    "What is BERT?",
    "How does attention work?",
    "Explain transformers",
]

for query in queries:
    with tracer.span(name="batch_search", input_data={"query": query}) as span:
        results = search_service.search(query=query, top_k=5)
        if span:
            span.event(name="results", output={"hit_count": len(results["text_hits"])})

tracer.flush()
```

### Example 3: Using Tracing Manager in FastAPI

The tracing is automatically integrated in server.py endpoints, but you can extend it:

```python
from fastapi import FastAPI
from backend.graphrag.tracing import get_tracing_manager

app = FastAPI()
tracer = get_tracing_manager()

@app.get("/my-endpoint")
async def my_endpoint(query: str):
    with tracer.trace(name="my_endpoint", input_data={"query": query}):
        result = do_something(query)
        return result
```

## Best Practices

1. **Name Traces Clearly**: Use descriptive names like "entity_extraction", "query_synthesis"
2. **Include Metadata**: Log parameters, model names, confidence scores
3. **Flush Regularly**: Call `tracer.flush()` after batch processing
4. **Monitor Errors**: Check error events in LangFuse dashboard
5. **Set Up Alerts**: Configure LangFuse alerts for high error rates or latency spikes
6. **Tag Experiments**: Add experiment tags to compare different model configs

## Advanced Configuration

## Advanced Configuration

### Self-Hosted LangFuse

For keeping data on your infrastructure, deploy LangFuse yourself.

#### Docker Compose Setup (Recommended)

```bash
mkdir -p langfuse-deployment && cd langfuse-deployment

cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  langfuse:
    image: langfuse/langfuse:latest
    ports:
      - "3000:3000"
    environment:
      # Database configuration
      DATABASE_URL: postgresql://postgres:changeme@postgres:5432/langfuse
      # Authentication secrets
      NEXTAUTH_SECRET: your-random-secret-key-here-change-this
      NEXTAUTH_URL: http://localhost:3000
      # For production with HTTPS
      # NEXTAUTH_URL: https://langfuse.yourdomain.com
    depends_on:
      - postgres
    restart: unless-stopped
    volumes:
      - ./logs:/app/logs

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: changeme
      POSTGRES_DB: langfuse
    ports:
      - "5432:5432"
    restart: unless-stopped
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups

volumes:
  postgres_data:
EOF

# Generate secure secrets
NEXTAUTH_SECRET=$(openssl rand -base64 32)
echo "NEXTAUTH_SECRET=$NEXTAUTH_SECRET" > .env

# Start services
docker-compose up -d

# Check status
docker-compose ps
```

Access LangFuse at: http://localhost:3000

#### Kubernetes Deployment

For production Kubernetes clusters:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: langfuse-config
data:
  DATABASE_URL: "postgresql://postgres:password@postgres:5432/langfuse"
  NEXTAUTH_URL: "https://langfuse.yourdomain.com"

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langfuse
spec:
  replicas: 2
  selector:
    matchLabels:
      app: langfuse
  template:
    metadata:
      labels:
        app: langfuse
    spec:
      containers:
      - name: langfuse
        image: langfuse/langfuse:latest
        ports:
        - containerPort: 3000
        envFrom:
        - configMapRef:
            name: langfuse-config
        env:
        - name: NEXTAUTH_SECRET
          valueFrom:
            secretKeyRef:
              name: langfuse-secrets
              key: nextauth-secret
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
```

Deploy with:
```bash
kubectl apply -f langfuse-deployment.yaml
```

#### VM Deployment (Ubuntu/Debian)

```bash
#!/bin/bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Create application directory
mkdir -p /opt/langfuse
cd /opt/langfuse

# Create docker-compose.yml (see above)
# ... paste the docker-compose.yml content ...

# Start services
docker-compose up -d

# Set up log rotation
cat > /etc/logrotate.d/langfuse << 'LOGROTATE'
/opt/langfuse/logs/*.log {
  weekly
  rotate 4
  compress
  delaycompress
  missingok
}
LOGROTATE

# Set up automatic backups
cat > /opt/langfuse/backup.sh << 'BACKUP'
#!/bin/bash
BACKUP_DIR="/opt/langfuse/backups"
DATE=$(date +%Y%m%d_%H%M%S)
docker-compose exec -T postgres pg_dump -U postgres langfuse > "$BACKUP_DIR/langfuse_$DATE.sql"
find "$BACKUP_DIR" -name "langfuse_*.sql" -mtime +30 -delete
BACKUP

chmod +x /opt/langfuse/backup.sh

# Add to crontab for daily backups
(crontab -l 2>/dev/null; echo "0 2 * * * /opt/langfuse/backup.sh") | crontab -
```

#### SSL/TLS with Reverse Proxy

Set up HTTPS with Nginx:

```nginx
# /etc/nginx/sites-available/langfuse
upstream langfuse_backend {
    server 127.0.0.1:3000;
}

server {
    listen 80;
    server_name langfuse.yourdomain.com;
    
    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name langfuse.yourdomain.com;

    # SSL certificates (use Let's Encrypt)
    ssl_certificate /etc/letsencrypt/live/langfuse.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/langfuse.yourdomain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://langfuse_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

Enable and restart:
```bash
sudo ln -s /etc/nginx/sites-available/langfuse /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Set up SSL with Let's Encrypt
sudo certbot certonly --nginx -d langfuse.yourdomain.com
```

### GraphRAG Configuration for Self-Hosted

Once your self-hosted instance is running:

```bash
# .env file
LANGFUSE_PUBLIC_KEY=pk_xxxxxxxxxxxxx
LANGFUSE_SECRET_KEY=sk_xxxxxxxxxxxxx
LANGFUSE_HOST=http://localhost:3000              # Local development
# or
LANGFUSE_HOST=https://langfuse.yourdomain.com    # Production
```

Or export directly:
```bash
export LANGFUSE_HOST="https://langfuse.yourdomain.com"
export LANGFUSE_PUBLIC_KEY="pk_..."
export LANGFUSE_SECRET_KEY="sk_..."

python -m backend.graphrag.cli serve --port 8000
```

The tracing manager will automatically detect and use your self-hosted instance:
```
✓ LangFuse tracing enabled (host=https://langfuse.yourdomain.com)
```

### Storage Configuration

#### PostgreSQL Optimization

For production deployments, optimize PostgreSQL:

```sql
-- Connect to langfuse database
psql -U postgres -d langfuse << 'SQL'

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- Create indexes for better query performance
CREATE INDEX idx_traces_project_id ON traces(project_id);
CREATE INDEX idx_traces_timestamp ON traces(timestamp DESC);
CREATE INDEX idx_spans_trace_id ON spans(trace_id);
CREATE INDEX idx_observations_trace_id ON observations(trace_id);

-- Analyze tables
ANALYZE;
SQL
```

#### Backup Strategy

```bash
#!/bin/bash
# Daily backup script
BACKUP_DIR="/opt/langfuse/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
docker-compose exec -T postgres pg_dump -U postgres langfuse | gzip > "$BACKUP_DIR/langfuse_$DATE.sql.gz"

# Keep only last 30 days
find "$BACKUP_DIR" -name "langfuse_*.sql.gz" -mtime +30 -delete

# Optional: Upload to S3
# aws s3 cp "$BACKUP_DIR/langfuse_$DATE.sql.gz" s3://your-bucket/langfuse-backups/

echo "Backup completed: $BACKUP_DIR/langfuse_$DATE.sql.gz"
```

Set credentials to empty to disable tracing:

```bash
# Disable LangFuse (no traces sent)
unset LANGFUSE_PUBLIC_KEY
unset LANGFUSE_SECRET_KEY
```

### Custom Metadata

Add custom fields to traces via the metadata parameter:

```python
tracer.log_llm_call(
    name="extraction",
    model="gemini-2.0-flash",
    prompt="...",
    response="...",
    metadata={
        "experiment": "v2",
        "temperature": 0.1,
        "paper_domain": "NLP",
    },
)
```

## References

- [LangFuse Documentation](https://langfuse.com/docs)
- [LangFuse Python SDK](https://github.com/langfuse/langfuse-python)
- [Tracing Best Practices](https://langfuse.com/docs/tracing)

## Support

For issues with LangFuse integration:

1. Check [LangFuse Discord Community](https://discord.gg/7yS3ZhyyMG)
2. Review [GitHub Issues](https://github.com/langfuse/langfuse/issues)
3. Check logs: `tail -f app.log | grep -i langfuse`
