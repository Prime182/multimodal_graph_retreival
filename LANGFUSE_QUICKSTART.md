# LangFuse Self-Hosted Quick Start Guide

## 🚀 Quick Setup (5 minutes) - Self-Hosted

### Step 1: Deploy Self-Hosted LangFuse

**Option A: Docker Compose** (Recommended)

```bash
mkdir langfuse-selfhosted && cd langfuse-selfhosted

cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  langfuse:
    image: langfuse/langfuse:latest
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/langfuse
      - NEXTAUTH_SECRET=your-secret-key-here
      - NEXTAUTH_URL=http://localhost:3000
    depends_on:
      - postgres

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=langfuse
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
EOF

docker-compose up -d
```

Your LangFuse instance will be available at: http://localhost:3000

**Option B: Kubernetes**

See: https://langfuse.com/docs/deployment/self-hosted

**Option C: Docker Run** (Simple)

```bash
docker run \
  -p 3000:3000 \
  -e DATABASE_URL="postgresql://user:password@postgres:5432/langfuse" \
  langfuse/langfuse
```

### Step 2: Get Credentials from Self-Hosted Instance

1. Open http://localhost:3000 (or your server IP)
2. Create an account (first user becomes admin)
3. Create a new project
4. Copy **Public Key** and **Secret Key** from project settings

### Step 3: Configure for Self-Hosted

```bash
# .env file or export directly
export LANGFUSE_PUBLIC_KEY="pk_xxxxxxxxxxxxx"
export LANGFUSE_SECRET_KEY="sk_xxxxxxxxxxxxx"
export LANGFUSE_HOST="http://localhost:3000"    # Your self-hosted URL
```

**For production** (change localhost to your server):
```bash
export LANGFUSE_HOST="http://your-langfuse-server.com"
export LANGFUSE_HOST="https://langfuse.yourdomain.com"  # With HTTPS
```

### Step 4: Start Server with Self-Hosted LangFuse

```bash
source .venv/bin/activate
python -m backend.graphrag.cli serve --port 8000
# Should see: ✓ LangFuse tracing enabled (host=http://localhost:3000)
```

### Step 5: View Traces in Self-Hosted Dashboard

Open: http://localhost:3000  
Click "Traces" and start using GraphRAG!

---

## 📊 What Gets Traced

| Operation | Traces | Metrics |
|-----------|--------|---------|
| **Search** | `document_search` | Hit counts, query type |
| **Synthesis** | `query_synthesis` | Answer length, confidence |
| **Entity Extraction** | `extract_layer2` | Entity count by type |
| **Entity Merging** | `entity_canonicalization` | Merge count, mapping size |
| **API Calls** | `api_search_*`, `api_synthesize` | Request/response data |
| **LLM Calls** | `entity_extraction`, `synthesis` | Model, tokens, latency |

---

## 🔍 Common Operations

### View All Traces
```python
from backend.graphrag.tracing import get_tracing_manager

tracer = get_tracing_manager()
print(f"Tracing enabled: {tracer.enabled}")
```

### Manual Tracing
```python
from backend.graphrag.tracing import get_tracing_manager

tracer = get_tracing_manager()

with tracer.trace(name="my_operation", input_data={"query": "test"}):
    result = do_something()

tracer.flush()
```

### Disable Tracing
```bash
# Just don't set environment variables
unset LANGFUSE_PUBLIC_KEY
unset LANGFUSE_SECRET_KEY
unset LANGFUSE_HOST
```

---

## 🛠️ Troubleshooting

**Q: Can't connect to self-hosted instance**
```bash
# Check connection
curl http://localhost:3000/api/health

# Or for remote server
curl https://your-server.com/api/health
```

**Q: Wrong credentials**
```bash
# Get correct keys from self-hosted dashboard at Settings → API Keys
export LANGFUSE_PUBLIC_KEY="pk_..."
export LANGFUSE_SECRET_KEY="sk_..."
```

**Q: Traces not appearing**
```bash
# Verify LANGFUSE_HOST is set correctly
echo $LANGFUSE_HOST
# Should show: http://localhost:3000 (or your server)

# Restart server
python -m backend.graphrag.cli serve --port 8000
```

---

## 🚀 Production Self-Hosted Setup

### HTTPS with Reverse Proxy (Nginx)

```nginx
server {
    listen 443 ssl;
    server_name langfuse.yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Configure GraphRAG:
```bash
export LANGFUSE_HOST="https://langfuse.yourdomain.com"
```

### Docker with Persistent Storage

```yaml
services:
  langfuse:
    image: langfuse/langfuse:latest
    ports:
      - "3000:3000"
    environment:
      DATABASE_URL: postgresql://postgres:${DB_PASSWORD}@postgres:5432/langfuse
      NEXTAUTH_SECRET: ${NEXTAUTH_SECRET}
      NEXTAUTH_URL: https://langfuse.yourdomain.com
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
```

---

## 📚 More Information

- **Full Self-Hosted Guide**: [docs/LANGFUSE_SETUP.md](../docs/LANGFUSE_SETUP.md)
- **Self-Hosted Deployment**: https://langfuse.com/docs/deployment/self-hosted
- **LangFuse Docker**: https://hub.docker.com/r/langfuse/langfuse

---

## ✨ Self-Hosted Benefits

✓ Data stays on your infrastructure  
✓ No cloud provider dependency  
✓ Full control over the instance  
✓ Can use with private networks  
✓ Unlimited trace storage  
✓ Same features as cloud version  

---

**Ready?** Deploy LangFuse, set env vars, and start tracing! 🚀


## 📊 What Gets Traced

| Operation | Traces | Metrics |
|-----------|--------|---------|
| **Search** | `document_search` | Hit counts, query type |
| **Synthesis** | `query_synthesis` | Answer length, confidence |
| **Entity Extraction** | `extract_layer2` | Entity count by type |
| **Entity Merging** | `entity_canonicalization` | Merge count, mapping size |
| **API Calls** | `api_search_*`, `api_synthesize` | Request/response data |
| **LLM Calls** | `entity_extraction`, `synthesis` | Model, tokens, latency |

## 🔍 Common Operations

### View All Traces
```python
from backend.graphrag.tracing import get_tracing_manager

tracer = get_tracing_manager()
print(f"Tracing enabled: {tracer.enabled}")
```

### Manual Tracing
```python
from backend.graphrag.tracing import get_tracing_manager

tracer = get_tracing_manager()

with tracer.trace(name="my_operation", input_data={"query": "test"}):
    result = do_something()

tracer.flush()
```

### Disable Tracing
```bash
# Just don't set environment variables
unset LANGFUSE_PUBLIC_KEY
unset LANGFUSE_SECRET_KEY
```

## 🛠️ Troubleshooting

**Q: Traces not showing in dashboard**
```bash
# Check credentials are set
echo $LANGFUSE_PUBLIC_KEY
echo $LANGFUSE_SECRET_KEY

# Check connection
curl https://cloud.langfuse.com/api/health
```

**Q: How to see if tracing is working**
```bash
python -c "
from backend.graphrag.tracing import get_tracing_manager
tracer = get_tracing_manager()
print(f'Enabled: {tracer.enabled}')
"
```

**Q: Performance impact?**
- Disabled (no creds): 0% overhead
- Enabled: <1% overhead (async background threads)

## 📚 More Information

- **Full Setup Guide**: [docs/LANGFUSE_SETUP.md](../docs/LANGFUSE_SETUP.md)
- **Integration Details**: [docs/LANGFUSE_INTEGRATION.md](../docs/LANGFUSE_INTEGRATION.md)
- **LangFuse Docs**: https://langfuse.com/docs

## 💡 Pro Tips

1. **Tag Your Traces**: Add metadata to group related traces
   ```python
   tracer.log_llm_call(
       name="extraction",
       model="gemini",
       prompt="...",
       response="...",
       metadata={
           "experiment": "v2_with_better_prompts",
           "paper_domain": "NLP",
       },
   )
   ```

2. **Monitor Token Usage**: Check token consumption per operation
   ```python
   # See token_used field in synthesis traces
   ```

3. **Set Up Alerts**: In LangFuse dashboard, set alerts for:
   - High error rates
   - Slow operations (>5s)
   - Token usage spikes

4. **Export Data**: Use LangFuse API to export traces for analysis

## 🎯 Next Steps

1. ✅ Set up LangFuse credentials
2. ✅ Start the server
3. ✅ Make API requests
4. ✅ View traces in dashboard
5. 🔄 Analyze performance and optimize

---

**Need help?** Check the full [LANGFUSE_SETUP.md](../docs/LANGFUSE_SETUP.md) guide or visit [LangFuse Discord Community](https://discord.gg/7yS3ZhyyMG)
