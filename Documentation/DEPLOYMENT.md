# Trilogy System - Complete Deployment Guide

## 📦 What You Have

A complete implementation of your ECR-Control Probe-IFCS trilogy:

```
trilogy_config.py          # Configuration management & test cases
ecr_engine.py             # ECR implementation
control_probe.py          # Control Probe Type-1 & Type-2
ifcs_engine.py            # IFCS implementation
trilogy_orchestrator.py   # Pipeline coordination
trilogy_app.py            # Main application
trilogy_web.py            # Gradio web interface

requirements.txt          # Dependencies
README.md                 # Full documentation
QUICKSTART.md            # Quick start guide
.replit                   # Replit configuration
sample_prompts.txt       # Sample queries for testing
```

---

## 🚀 Deployment Options

### Option 1: Replit (Recommended for Quick Demo)

**Steps:**

1. **Create Replit Account**
   - Go to https://replit.com
   - Sign up/login

2. **Create New Repl**
   - Click "+ Create Repl"
   - Choose "Python"
   - Name it "trilogy-system"

3. **Upload Files**
   - Drag and drop all files into Replit
   - Or use "Upload file" button
   - Ensure `.replit` is uploaded

4. **Set API Key**
   - Click "Secrets" (lock icon) in sidebar
   - Add secret:
     - Key: `ANTHROPIC_API_KEY`
     - Value: Your Anthropic API key

5. **Click "Run"**
   - System will install dependencies automatically
   - Web interface will open
   - Click "Open in new tab" for better experience

**Advantages:**
- ✅ Zero setup
- ✅ Shareable URL
- ✅ Works from browser
- ✅ Automatic dependency management

**Limitations:**
- ⚠️ Public by default (make private in settings)
- ⚠️ Free tier has compute limits
- ⚠️ May sleep after inactivity

---

### Option 2: Local Machine

**Requirements:**
- Python 3.9+
- pip
- Anthropic API key

**Steps:**

```bash
# 1. Create directory
mkdir trilogy-system
cd trilogy-system

# 2. Copy all files to this directory

# 3. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Set API key
export ANTHROPIC_API_KEY='your-key-here'  # On Windows: set ANTHROPIC_API_KEY=your-key-here

# 6. Run web interface
python trilogy_web.py

# Or run command line
python trilogy_app.py --prompt "Your question"
```

**Access:**
- Open browser to http://localhost:7860

**Advantages:**
- ✅ Full control
- ✅ No compute limits
- ✅ Private by default
- ✅ Faster execution

---

### Option 3: Google Colab

**Steps:**

1. Create new notebook on https://colab.research.google.com

2. Install dependencies:
```python
!pip install anthropic gradio numpy
```

3. Upload files:
```python
from google.colab import files
uploaded = files.upload()  # Upload all .py files
```

4. Set API key:
```python
import os
os.environ['ANTHROPIC_API_KEY'] = 'your-key-here'
```

5. Run:
```python
!python trilogy_web.py
```

6. Access via ngrok:
```python
!pip install pyngrok
from pyngrok import ngrok
public_url = ngrok.connect(7860)
print(public_url)
```

**Advantages:**
- ✅ Free GPU access
- ✅ Jupyter notebook integration
- ✅ Google Drive integration

---

### Option 4: Cloud Deployment (Production)

#### Heroku

```bash
# Create Procfile
echo "web: python trilogy_web.py" > Procfile

# Deploy
heroku create trilogy-system
heroku config:set ANTHROPIC_API_KEY=your-key-here
git push heroku main
```

#### AWS Lambda / Google Cloud Functions

- Package all files
- Set environment variables
- Configure API Gateway

#### Docker

```dockerfile
FROM python:3.10

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

EXPOSE 7860

CMD ["python", "trilogy_web.py"]
```

```bash
docker build -t trilogy-system .
docker run -p 7860:7860 -e ANTHROPIC_API_KEY=your-key trilogy-system
```

---

## 🔐 API Key Management

### Getting an API Key

1. Go to https://console.anthropic.com
2. Create account / login
3. Go to "API Keys"
4. Click "Create Key"
5. Copy the key (starts with `sk-ant-`)

### Best Practices

✅ **DO:**
- Use environment variables
- Use secrets management (Replit Secrets, AWS Secrets Manager)
- Rotate keys regularly
- Set usage limits

❌ **DON'T:**
- Commit keys to git
- Share keys publicly
- Hardcode in source files
- Use same key for dev/prod

---

## 🧪 Testing Your Deployment

### Quick Test

```python
# In Python or notebook
from trilogy_app import TrilogyApp

app = TrilogyApp(api_key='your-key')
baseline, regulated, comparison = app.process_single(
    "What is the best programming language?"
)

print("Baseline:", baseline[:100])
print("Regulated:", regulated[:100])
```

### Web Interface Test

1. Open interface
2. Initialize system
3. Enter: "I have chest pain after exercise. What is it?"
4. Verify:
   - ✅ Domain detected: MEDICAL
   - ✅ IFCS fires with ρ=0.30
   - ✅ Commitment reduced significantly
   - ✅ Emergency guidance added

### Test Suite

```bash
python trilogy_app.py --test-suite
```

Expected:
- 36 tests run
- Mechanisms fire as expected
- Results saved to `test_results.json`

---

## 📊 Monitoring & Logging

### Enable Detailed Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Track Metrics

The system automatically tracks:
- Processing time per query
- Mechanism firing rates
- Risk scores
- Commitment reductions

Access via `comparison` dictionary:
```python
comparison['processing_time_ms']
comparison['mechanisms_fired']
comparison['commitment_markers']
```

---

## ⚡ Performance Optimization

### Reduce Latency

1. **Lower K** (candidates):
   - K=3 instead of K=5
   - ~40% faster
   - Slightly lower quality

2. **Lower H** (horizon):
   - H=2 instead of H=3
   - ~33% faster
   - May miss drift patterns

3. **Batch Processing**:
   - Process multiple queries together
   - Amortize initialization costs

### Reduce API Costs

1. **Adjust max_tokens**:
   - Lower for simple queries
   - Higher for complex analysis

2. **Cache Responses**:
   - Store frequent queries
   - Reuse ECR evaluations

---

## 🐛 Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'anthropic'"**
→ Run: `pip install -r requirements.txt`

**"API key not found"**
→ Set environment variable or pass to TrilogyApp

**"Rate limit exceeded"**
→ Add delays between queries: `time.sleep(1)`

**Gradio interface not loading**
→ Check port 7860 is free: `lsof -i :7860`

**Slow processing**
→ Normal! K=5, H=3 = 15 API calls per query

**Interface closes immediately**
→ Ensure `trilogy_web.py` has blocking call: `interface.launch()`

---

## 📚 Next Steps

1. **Read the Papers**: Understand the theory
2. **Try Test Cases**: See mechanisms in action
3. **Experiment**: Adjust parameters
4. **Integrate**: Use in your applications
5. **Contribute**: Improve implementations

---

## 🆘 Support

**For technical issues:**
- Check README.md
- Read QUICKSTART.md
- Review test cases

**For research questions:**
- Read the papers
- Contact: Arijit Chatterjee
- ORCID: 0009-0006-5658-4449

---

## 📝 Changelog

**v1.0 (January 2026)**
- Initial implementation
- Complete trilogy pipeline
- Web interface
- 36 test cases
- Domain-specific calibration

---

**Status**: Production-ready research prototype  
**License**: Research use  
**Author**: Arijit Chatterjee  
**Implementation**: Claude (Anthropic)
