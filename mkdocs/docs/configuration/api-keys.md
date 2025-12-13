# API Keys Configuration

Panduan lengkap setup dan konfigurasi API keys yang diperlukan.

---

## 🔑 Required API Keys

| Service      | Purpose                       | Required | Free Tier                    | Usage                     |
| ------------ | ----------------------------- | -------- | ---------------------------- | ------------------------- |
| Hugging Face | LLM (Llama 3.1-8B) Assessment | ✅ Yes   | ✅ Unlimited (Inference API) | ~15-30s per question      |
| DeepL        | Translation (EN↔ID)           | ✅ Yes   | ✅ 500k chars/month          | ~100-500 chars per answer |

**Estimated Monthly Usage (100 interviews, 3 questions each):**

- **Hugging Face**: 300 API calls (FREE - no credit card needed)
- **DeepL**: ~50,000 characters (10% of free tier)

**Total Cost: $0/month** ✅

---

## 🚀 Quick Setup

### 1. Create `.env` File

Buat file `.env` di root folder project:

```bash
d:\Interview_Assesment_System-main\backend\Python\.env
```

### 2. Add API Keys

```env
# Hugging Face API Token (Required)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxx

# DeepL API Key (Required)
DEEPL_API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx:fx
```

**⚠️ Important:** Variable names harus exact seperti di atas:

- `HF_TOKEN` (bukan `HUGGINGFACE_API_KEY`)
- `DEEPL_API_KEY` (bukan `DEEPL_AUTH_KEY`)

---

## 🤗 Hugging Face Setup

### Step 1: Create Account

1. Kunjungi [https://huggingface.co/join](https://huggingface.co/join)
2. Sign up dengan email atau GitHub
3. Verify email address

### Step 2: Get API Key

1. Login ke Hugging Face
2. Go to **Settings** → **Access Tokens**
3. Click **New token**
4. Token name: `interview-assessment`
5. Role: **Read**
6. Click **Generate token**
7. **Copy token** (starts with `hf_...`)

### Step 3: Add to `.env`

```env
HUGGINGFACE_API_KEY=hf_abcdefghijklmnopqrstuvwxyz1234567890
```

### Test API Key

```python
from transformers import pipeline

# Test connection
pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.1-8B-Instruct",
    token="hf_your_token_here"
)

print("✅ Hugging Face API working!")
```

---

## 🌐 DeepL Setup

### Step 1: Create Account

1. Kunjungi [https://www.deepl.com/pro-api](https://www.deepl.com/pro-api)
2. Click **Sign up for free**
3. Fill in details
4. Verify email

### Step 2: Get API Key

1. Login to DeepL account
2. Go to **Account** → **API Keys**  
   URL: [https://www.deepl.com/account/summary](https://www.deepl.com/account/summary)
3. Copy **Authentication Key for DeepL API**
4. Format: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx:fx` (ends with `:fx` for free tier)

**Note:** Free tier requires credit card verification but **no charges** until you upgrade.

### Step 3: Add to `.env`

```env
DEEPL_API_KEY=12345678-90ab-cdef-1234-567890abcdef:fx
```

### Test API Key

```python
import os
from dotenv import load_dotenv
import deepl

# Load environment
load_dotenv()
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")

if not DEEPL_API_KEY:
    raise ValueError("❌ DEEPL_API_KEY not found in .env")

# Test translation
translator = deepl.Translator(DEEPL_API_KEY)

try:
    # Test EN → ID
    result_id = translator.translate_text(
        "Hello, how are you?",
        target_lang="ID"
    )
    print(f"EN → ID: {result_id.text}")

    # Test ID → EN
    result_en = translator.translate_text(
        "Halo, apa kabar?",
        source_lang="ID",
        target_lang="EN-US"
    )
    print(f"ID → EN: {result_en.text}")

    # Check usage
    usage = translator.get_usage()
    print(f"\n✅ DeepL API working!")
    print(f"Usage: {usage.character.count:,} / {usage.character.limit:,} characters")
    print(f"Remaining: {usage.character.limit - usage.character.count:,} characters")

except Exception as e:
    print(f"❌ Error: {e}")
```

**Expected Output:**

```
EN → ID: Halo, apa kabar?
ID → EN: Hello, how are you?

✅ DeepL API working!
Usage: 45 / 500,000 characters
Remaining: 499,955 characters
```

### Free Tier Limits

| Plan | Characters/Month | Cost                 |
| ---- | ---------------- | -------------------- |
| Free | 500,000          | $0                   |
| Pro  | Unlimited        | Starting $5.99/month |

---

## 📝 Environment File Examples

### Complete `.env` File

```env
# ==================================================
# AI Interview Assessment System - Environment Config
# ==================================================

# Hugging Face API (Required)
# Get from: https://huggingface.co/settings/tokens
HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz1234567890ABCD

# DeepL API (Required)
# Get from: https://www.deepl.com/account/summary
DEEPL_API_KEY=12345678-90ab-cdef-1234-567890abcdef:fx
```

### With Comments (`.env.example`)

```env
# ==================================================
# AI Interview Assessment System - Configuration
# ==================================================

# Hugging Face Inference API Token
# - Used for: Llama 3.1-8B-Instruct LLM assessment
# - Free tier: Unlimited requests (no credit card)
# - Get token: https://huggingface.co/settings/tokens
# - Permissions: Read access only
HF_TOKEN=hf_your_token_here

# DeepL Translation API Key
# - Used for: EN ↔ ID translation
# - Free tier: 500,000 characters/month
# - Get key: https://www.deepl.com/account/summary
# - Note: Ends with ':fx' for free tier
DEEPL_API_KEY=your_deepl_key_here:fx
```

---

## 🔐 Security Best Practices

### ✅ DO

- **Store keys in `.env`** file
- **Add `.env` to `.gitignore`**
- **Use different keys for dev/prod**
- **Rotate keys regularly** (every 90 days)
- **Restrict key permissions** (read-only when possible)

### ❌ DON'T

- ❌ Commit `.env` to Git
- ❌ Share keys in chat/email
- ❌ Hardcode keys in source code
- ❌ Use production keys in development
- ❌ Store keys in public repositories

---

## 📋 `.gitignore` Configuration

Pastikan `.env` ada di `.gitignore`:

```gitignore
# Environment variables
.env
.env.local
.env.development
.env.production

# API keys
*.key
secrets/
```

---

## 🔄 Loading Environment Variables

### In Jupyter Notebook

**Cell 1: Load Environment Variables**

```python
import os
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

# Access variables
HF_TOKEN = os.getenv('HF_TOKEN')
DEEPL_API_KEY = os.getenv('DEEPL_API_KEY')

# Verify required keys
if not HF_TOKEN:
    raise ValueError("❌ HF_TOKEN not found in .env")
if not DEEPL_API_KEY:
    raise ValueError("❌ DEEPL_API_KEY not found in .env")

print("✅ Environment variables loaded")
print(f"HF_TOKEN: {HF_TOKEN[:10]}...{HF_TOKEN[-4:]}")
print(f"DEEPL_API_KEY: {DEEPL_API_KEY[:8]}...{DEEPL_API_KEY[-3:]}")
```

**Expected Output:**

```
✅ Environment variables loaded
HF_TOKEN: hf_abcdefg...ABCD
DEEPL_API_KEY: 12345678....:fx
```

````

### In Python Script

```python
#!/usr/bin/env python
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Access API keys
HF_TOKEN = os.getenv('HF_TOKEN')
DEEPL_API_KEY = os.getenv('DEEPL_API_KEY')

# Validate
if not HF_TOKEN:
    raise ValueError("❌ HF_TOKEN not found")
if not DEEPL_API_KEY:
    raise ValueError("❌ DEEPL_API_KEY not found")

print("✅ All API keys loaded")
````

---

## 🧪 Validation Script

Create `check_api_keys.py`:

```python
#!/usr/bin/env python
"""
Validate all API keys are configured correctly
"""
import os
import sys
from dotenv import load_dotenv

def check_api_keys():
    load_dotenv()

    required_keys = {
        'HF_TOKEN': 'Hugging Face',
        'DEEPL_API_KEY': 'DeepL'
    }

    all_ok = True

    print("🔍 Checking Required API Keys...\n")

    # Check required keys
    for key, service in required_keys.items():
        value = os.getenv(key)
        if value and len(value) > 10:
            print(f"✅ {service}: Configured")
        else:
            print(f"❌ {service}: Missing or invalid")
            all_ok = False

    print("\n" + "="*50)

    if all_ok:
        print("✅ All required API keys are configured!")
        return 0
    else:
        print("❌ Some required API keys are missing!")
        print("\n📝 Please check your .env file")
        return 1

if __name__ == "__main__":
    sys.exit(check_api_keys())
```

Run validation:

```bash
python check_api_keys.py
```

---

## 🔧 Troubleshooting

### Error: "API key not found"

**Solution:**

```bash
# Check .env exists
ls -la .env

# Check .env content (safely)
cat .env | sed 's/=.*/=***HIDDEN***/'

# Reload environment
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('HF_TOKEN:', os.getenv('HF_TOKEN')[:10] if os.getenv('HF_TOKEN') else 'NOT FOUND')"
```

---

### Error: "Invalid API key"

**Hugging Face:**

- Key harus start dengan `hf_`
- Check di https://huggingface.co/settings/tokens

**DeepL:**

- Key harus end dengan `:fx`
- Check di https://www.deepl.com/account

---

### Error: "Rate limit exceeded"

**Solutions:**

1. **Wait and retry:**

   ```python
   import time
   time.sleep(60)  # Wait 1 minute
   ```

2. **Upgrade plan:**

   - Hugging Face: Pro plan
   - DeepL: Pro API

3. **Use caching:**

   ```python
   # Cache translations
   translation_cache = {}

   def translate_cached(text, target_lang):
       key = f"{text}_{target_lang}"
       if key not in translation_cache:
           translation_cache[key] = translator.translate_text(text, target_lang=target_lang)
       return translation_cache[key]
   ```

---

## 📊 API Usage Monitoring

### Verify Hugging Face Token

```python
import os
import requests
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def check_hf_token(token):
    """Verify HF token validity."""
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(
            "https://huggingface.co/api/whoami-v2",
            headers=headers
        )
        if response.status_code == 200:
            data = response.json()
            print("✅ Hugging Face Token Valid")
            print(f"Username: {data.get('name', 'N/A')}")
            print(f"Type: {data.get('type', 'N/A')}")
            return True
        else:
            print(f"❌ Invalid token: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# Test
check_hf_token(HF_TOKEN)
```

````

### Track DeepL Usage

```python
import os
from dotenv import load_dotenv
import deepl

load_dotenv()
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")

translator = deepl.Translator(DEEPL_API_KEY)
usage = translator.get_usage()

print(f"📊 DeepL Usage Statistics:")
print(f"Character count: {usage.character.count:,}")
print(f"Character limit: {usage.character.limit:,}")
print(f"Remaining: {usage.character.limit - usage.character.count:,}")

# Calculate percentage
percentage_used = (usage.character.count / usage.character.limit) * 100
print(f"Usage: {percentage_used:.2f}%")

# Warning if approaching limit
if percentage_used > 80:
    print("⚠️ Warning: Approaching character limit!")
elif percentage_used > 50:
    print("ℹ️ Info: Over 50% quota used")
else:
    print("✅ Plenty of quota remaining")
````

**Expected Output:**

```
📊 DeepL Usage Statistics:
Character count: 12,345
Character limit: 500,000
Remaining: 487,655
Usage: 2.47%
✅ Plenty of quota remaining
```

---

## 💡 Cost Optimization Tips

1. **Cache translations** - Avoid duplicate API calls for same text

   ```python
   translation_cache = {}

   def translate_cached(text, target_lang):
       cache_key = f"{text}_{target_lang}"
       if cache_key not in translation_cache:
           translation_cache[cache_key] = translator.translate_text(
               text, target_lang=target_lang
           )
       return translation_cache[cache_key]
   ```

2. **Monitor DeepL usage** - Check quota regularly

   ```python
   usage = translator.get_usage()
   if usage.character.count > 400000:  # 80% of 500k
       print("⚠️ Approaching DeepL quota limit!")
   ```

3. **Optimize LLM prompts** - Shorter prompts = faster responses

   ```python
   # Good: Concise prompt
   prompt = "Assess this answer (0-100): {answer}"

   # Avoid: Overly verbose prompts
   # prompt = "Please carefully analyze and thoroughly evaluate..."
   ```

4. **Use free tiers effectively**:
   - Hugging Face: Unlimited free requests ✅
   - DeepL: 500k chars/month (≈400 interviews) ✅
   - Total: **$0/month for moderate usage**

---

## 🔄 Complete Setup Workflow

**Step-by-step guide from scratch:**

```bash
# 1. Navigate to project
cd d:\Coding\CheatingDetection\Interview\Interview_Assesment_System

# 2. Create .env file
touch .env  # Linux/Mac
# or
New-Item .env -ItemType File  # Windows PowerShell

# 3. Open .env in editor
notepad .env  # Windows
# or
nano .env     # Linux/Mac
```

**Add to .env:**

```env
HF_TOKEN=hf_your_token_here
DEEPL_API_KEY=your_deepl_key_here:fx
```

**Test setup:**

```python
# Run in Jupyter notebook or Python
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import deepl

# Load
load_dotenv()

# Test HF
hf_token = os.getenv("HF_TOKEN")
client = InferenceClient(api_key=hf_token)
response = client.text_generation(
    model="meta-llama/Llama-3.1-8B-Instruct",
    prompt="Hello",
    max_new_tokens=5
)
print(f"✅ HF: {response}")

# Test DeepL
deepl_key = os.getenv("DEEPL_API_KEY")
translator = deepl.Translator(deepl_key)
result = translator.translate_text("Test", target_lang="ID")
print(f"✅ DeepL: {result.text}")

print("\n🎉 All API keys configured successfully!")
```

---

---

## ✅ Quick Validation Checklist

Before running the system, verify:

- [ ] `.env` file created in project root
- [ ] `HF_TOKEN` added (starts with `hf_`, ~37 chars)
- [ ] `DEEPL_API_KEY` added (ends with `:fx`)
- [ ] `.env` added to `.gitignore`
- [ ] Both API keys tested successfully
- [ ] DeepL quota checked (should have 500k characters)
- [ ] Hugging Face token verified (read access)

## 🚀 Next Steps

After configuring API keys:

1. **Test System**: Run [quickstart guide](../getting-started/quickstart.md)
2. **Configure Models**: See [models.md](models.md) for model settings
3. **API Testing**: Try [endpoints](../api/endpoints.md)
4. **Troubleshooting**: Check [common issues](../troubleshooting/common-issues.md) if errors occur

---

## 📚 Additional Resources

- [Hugging Face Inference API Docs](https://huggingface.co/docs/api-inference/index)
- [DeepL API Documentation](https://www.deepl.com/docs-api)
- [Model Configuration Guide](models.md)
- [Advanced Settings](advanced.md)
- [Troubleshooting Guide](../troubleshooting/common-issues.md)
