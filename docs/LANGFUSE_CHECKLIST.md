# LangFuse Integration Checklist & Reference

## 🎯 Pre-Integration Checklist

### Before You Start
- [ ] LangFuse account created (https://cloud.langfuse.com)
- [ ] Project created in LangFuse
- [ ] Public Key copied (pk_prod_xxxx)
- [ ] Secret Key copied (sk_prod_xxxx)
- [ ] GraphRAG project requirements installed (`pip install -r requirements.txt`)

---

## 📋 Setup Checklist

### Step 1: Environment Configuration
```bash
# Choose your method:

# Option A: .env file (recommended)
cat >> .env << 'EOF'
LANGFUSE_PUBLIC_KEY=pk_prod_xxxxxxxxxxxxx
LANGFUSE_SECRET_KEY=sk_prod_xxxxxxxxxxxxx
LANGFUSE_HOST=https://cloud.langfuse.com
EOF
source .env

# Option B: Export directly
export LANGFUSE_PUBLIC_KEY="pk_prod_xxxxxxxxxxxxx"
export LANGFUSE_SECRET_KEY="sk_prod_xxxxxxxxxxxxx"

# Option C: System environment (permanent)
# Add to ~/.bashrc or ~/.zshrc
```

- [ ] Environment variables set
- [ ] Verified with: `echo $LANGFUSE_PUBLIC_KEY`

### Step 2: Install Dependencies
```bash
source .venv/bin/activate
pip install -r requirements.txt
# Or just: pip install langfuse>=2.0,<3
```

- [ ] LangFuse package installed (`pip list | grep langfuse`)
- [ ] OpenTelemetry installed (`pip list | grep opentelemetry`)

### Step 3: Verify Installation
```bash
python -c "
from backend.graphrag.tracing import get_tracing_manager
tracer = get_tracing_manager()
print(f'✓ Tracing enabled: {tracer.enabled}')
"
```

- [ ] No import errors
- [ ] Shows `Tracing enabled: True` (if credentials valid)

---

## 🚀 Running with Tracing

### Server Startup
```bash
# Activate environment
source .venv/bin/activate

# Start server
python -m backend.graphrag.cli serve --port 8000
# You should see: ✓ LangFuse tracing enabled (host=https://cloud.langfuse.com)
```

- [ ] Server starts without errors
- [ ] See "LangFuse tracing enabled" message
- [ ] Navigate to http://localhost:8000/docs for OpenAPI docs

### Testing Traces
```bash
# Make a search request
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "BERT transformer", "top_k": 5}'

# Make a synthesis request (RAG)
curl -X POST http://localhost:8000/api/synthesize \
  -H "Content-Type: application/json" \
  -d '{"query": "What is BERT?", "top_k": 5}'
```

- [ ] Requests complete successfully (HTTP 200)
- [ ] No errors in server logs

### Viewing Traces
1. [ ] Open https://cloud.langfuse.com/dashboard
2. [ ] Click "Traces" in left sidebar
3. [ ] Look for recent traces:
   - [ ] `document_search` traces
   - [ ] `query_synthesis` traces (if synthesis endpoint called)
   - [ ] `api_search_*` traces
   - [ ] `api_synthesize` traces

---

## 📊 What You Should See

### In LangFuse Dashboard

```
Traces (Recent):
├── document_search (api_search_post)
│   ├── Input: {"query": "BERT transformer", "top_k": 5}
│   ├── Output: {"text_hits": 5, "papers": 3, "entities": 12}
│   └── Duration: 245ms
│
├── api_synthesize
│   ├── Input: {"query": "What is BERT?"}
│   ├── query_synthesis (nested)
│   │   ├── synthesis (LLM call)
│   │   ├── Tokens: input:450, output:120
│   │   └── Duration: 3.2s
│   └── Output: {"answer_length": "238", "confidence": "0.92"}
│
└── extract_layer2 (background)
    ├── entity_extraction (Gemini LLM)
    ├── Output: {"entity_count": 45, "entity_types": {...}}
    └── Duration: 8.5s
```

---

## 🔍 Monitoring Traces

### Common Queries in LangFuse

| What | How |
|------|-----|
| **See all searches** | Filter by trace name: `document_search` |
| **See all synthesis** | Filter by trace name: `query_synthesis` or `api_synthesize` |
| **See all errors** | Filter by trace name contains: `error` |
| **See slow operations** | Sort by duration, descending |
| **See token usage** | Look for `input` and `output` tokens in metadata |

### Dashboard Navigation

1. **Traces Tab**: View all operations
   - Filter by trace name
   - Filter by time range
   - Sort by duration, creation time, etc.

2. **Click on Trace**: See details
   - Full request/response
   - Nested spans
   - Token usage
   - Timestamps
   - Metadata

3. **Export Data**: Export traces for analysis
   - Export to CSV
   - Export to JSON

---

## 🛠️ Troubleshooting

### Issue: "Tracing disabled. Set credentials..."
**Solution**:
```bash
# Check if credentials are set
echo "Public: $LANGFUSE_PUBLIC_KEY"
echo "Secret: $LANGFUSE_SECRET_KEY"

# If empty, set them
export LANGFUSE_PUBLIC_KEY="pk_..."
export LANGFUSE_SECRET_KEY="sk_..."

# Restart server
```

- [ ] Credentials confirmed set
- [ ] Server restarted

### Issue: Traces not appearing in dashboard
**Checklist**:
1. [ ] Credentials are correct format (`pk_prod_*` and `sk_prod_*`)
2. [ ] Network connectivity to LangFuse (try: `curl https://cloud.langfuse.com/api/health`)
3. [ ] Project exists in LangFuse dashboard
4. [ ] Wait 5-10 seconds for traces to appear (slight delay)
5. [ ] Check server logs for errors (look for "error" keyword)

### Issue: High memory usage
**Likely cause**: Many traces pending flush  
**Solution**:
```python
from backend.graphrag.tracing import get_tracing_manager
tracer = get_tracing_manager()
tracer.flush()  # Explicitly flush traces
```

### Issue: API responses slow
**Unlikely**: Tracing overhead <1%  
**Check**:
```bash
# Test without tracing
unset LANGFUSE_PUBLIC_KEY
unset LANGFUSE_SECRET_KEY
# Restart server and measure response time
```

---

## 📚 Documentation References

| Document | Where | Purpose |
|----------|-------|---------|
| **LANGFUSE_QUICKSTART.md** | Root | 5-minute setup |
| **docs/LANGFUSE_SETUP.md** | docs/ | Complete guide |
| **docs/LANGFUSE_INTEGRATION.md** | docs/ | Technical details |
| **docs/LANGFUSE_IMPLEMENTATION.md** | docs/ | Summary & reference |

---

## 🎓 Learning Path

### Beginner (Just want tracing working)
1. Read: [LANGFUSE_QUICKSTART.md](../LANGFUSE_QUICKSTART.md)
2. Set credentials
3. Run server
4. Check dashboard

### Intermediate (Want to understand what's traced)
1. Read: [docs/LANGFUSE_IMPLEMENTATION.md](LANGFUSE_IMPLEMENTATION.md)
2. Learn trace point names
3. Create custom searches in LangFuse

### Advanced (Want to add custom tracing)
1. Read: [docs/LANGFUSE_SETUP.md](LANGFUSE_SETUP.md) - "Manual Tracing" section
2. Look at examples in [extraction.py](../backend/graphrag/extraction.py)
3. Implement custom `tracer.trace()` blocks

---

## 💡 Pro Tips

### Tip 1: Organize Traces with Tags
```python
tracer.log_llm_call(
    name="extraction",
    model="gemini",
    prompt="...",
    response="...",
    metadata={
        "experiment": "v2_optimized_prompts",
        "team": "nlp",
        "version": "1.2",
    },
)
```

### Tip 2: Monitor Specific Papers
```python
tracer.log_entity_extraction(
    paper_id="arxiv_2312.10997",
    entity_count=42,
    entity_types={"Concept": 25, "Method": 17},
)
```

### Tip 3: Track Across Pipeline Stages
Use consistent `metadata` fields to track items through:
- Extraction → Search → Synthesis

### Tip 4: Set Up Dashboard Alerts
In LangFuse: Settings → Alerts
- Alert on error rate > 5%
- Alert on latency > 5s
- Alert on trace count spike

### Tip 5: Export for Analysis
Use LangFuse API to export traces and build custom dashboards:
```bash
# Example: Export last 1000 traces
curl https://cloud.langfuse.com/api/traces \
  -H "Authorization: Bearer $LANGFUSE_SECRET_KEY" \
  -d '{"limit": 1000}' > traces.json
```

---

## ✅ Verification Checklist

### After Setup
- [ ] Environment variables set and verified
- [ ] LangFuse packages installed
- [ ] Credentials are valid (check in LangFuse dashboard)
- [ ] Server starts with "tracing enabled" message

### After Running
- [ ] Made at least one API request to server
- [ ] Waited 10 seconds
- [ ] Traces appear in LangFuse dashboard
- [ ] Can click on trace to see details
- [ ] Can see input/output data

### Performance Check
- [ ] Server response time normal (<500ms for searches)
- [ ] No memory leaks (memory stable over time)
- [ ] No error messages in logs

---

## 🚨 Quick Debug Commands

```bash
# Verify credentials
echo "Public Key: ${LANGFUSE_PUBLIC_KEY:0:20}..."
echo "Secret Key: ${LANGFUSE_SECRET_KEY:0:20}..."

# Check LangFuse connectivity
curl -I https://cloud.langfuse.com

# Test Python integration
python -c "
from langfuse import Langfuse
from backend.graphrag.tracing import get_tracing_manager
trace = get_tracing_manager()
print(f'Enabled: {trace.enabled}')
print(f'LangFuse: {trace.langfuse}')
"

# View recent traces (requires jq)
curl https://cloud.langfuse.com/api/traces \
  -H "Authorization: Bearer YOUR_SECRET_KEY" | jq '.data | length'

# Check server logs for errors
tail -f /tmp/graphrag_server.log | grep -i "langfuse\|error"
```

---

## 📞 Getting Help

### If Something Breaks
1. Check this checklist
2. Read [docs/LANGFUSE_SETUP.md](LANGFUSE_SETUP.md) - Troubleshooting section
3. Visit [LangFuse Discord](https://discord.gg/7yS3ZhyyMG)
4. Check [LangFuse GitHub Issues](https://github.com/langfuse/langfuse/issues)

### Quick Links
- LangFuse Dashboard: https://cloud.langfuse.com/dashboard
- LangFuse Docs: https://langfuse.com/docs
- GraphRAG Repo: (your repo)
- Discord: https://discord.gg/7yS3ZhyyMG

---

## 🎉 You're All Set!

Once you see traces in the LangFuse dashboard, you have:
✅ Real-time monitoring of all GraphRAG operations  
✅ Token usage tracking per operation  
✅ Error tracking and debugging  
✅ Performance analytics  
✅ Source attribution for synthesis  

**Happy tracing!** 🚀
