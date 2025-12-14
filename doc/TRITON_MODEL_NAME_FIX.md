# Triton Model Name Fix

## Issue

**Error:**
```
Thread [15] had error: OpenAI response returns HTTP code 400: 
{"detail":"Unknown model: Qwen/Qwen3-30B-A3B-Thinking-2507"}
```

## Root Cause

**Mismatch between model name formats:**

1. **HuggingFace format**: `Qwen/Qwen3-30B-A3B-Thinking-2507` (with slash)
2. **Triton model repository**: `Qwen_Qwen3-30B-A3B-Thinking-2507` (with underscore)

When deploying to Triton, the model repository structure uses underscores to replace slashes because:
- File system paths don't allow slashes in directory names
- Triton expects model names that are valid directory names

## The Problem

```python
# In docker.py (Triton deployment)
model_name = self.model.replace('/', '_')  # Qwen_Qwen3-30B-A3B-Thinking-2507
model_dir = model_repo_dir / model_name

# In measure.py (GenAI-Perf request) - BEFORE FIX
cmd_parts.append(f"-m {self.model}")  # Qwen/Qwen3-30B-A3B-Thinking-2507
# ❌ Triton doesn't recognize this name!
```

**What happens:**
1. Triton model repository has: `/models/Qwen_Qwen3-30B-A3B-Thinking-2507/`
2. GenAI-Perf sends request with: `model="Qwen/Qwen3-30B-A3B-Thinking-2507"`
3. Triton looks for: `/models/Qwen/Qwen3-30B-A3B-Thinking-2507/` ❌ Not found!
4. Returns: `HTTP 400: Unknown model`

## Solution

Detect when using Triton method and sanitize the model name for API requests:

```python
# In measure.py
model_name_for_request = self.model
if self.method == "triton":
    model_name_for_request = self.model.replace('/', '_')

cmd_parts.append(f"-m {model_name_for_request}")
```

Now:
- Triton gets: `Qwen_Qwen3-30B-A3B-Thinking-2507` ✅
- Matches repository: `/models/Qwen_Qwen3-30B-A3B-Thinking-2507/` ✅

## Key Insight

Different deployment methods have different naming conventions:

| Method | Model Name Format | Example |
|--------|-------------------|---------|
| HF | Original with `/` | `Qwen/Qwen3-30B-A3B-Thinking-2507` |
| Triton | Replace `/` with `_` | `Qwen_Qwen3-30B-A3B-Thinking-2507` |
| NIM | TBD | TBD |
| UNIM | TBD | TBD |

## Implementation Details

### In `measure.py`:

```python
def _build_genai_perf_command(self, mode):
    # Prepare model name based on deployment method
    model_name_for_request = self.model
    if self.method == "triton":
        # Triton uses underscores in model repository
        model_name_for_request = self.model.replace('/', '_')
    
    # Use sanitized name for API requests
    cmd_parts.append(f"-m {model_name_for_request}")
    
    # But keep original for tokenizer
    cmd_parts.append(f"--tokenizer {self.model}")
```

### Why Keep Original for Tokenizer?

The `--tokenizer` argument still uses the original HuggingFace model name because:
1. GenAI-Perf downloads tokenizer directly from HuggingFace Hub
2. HuggingFace Hub uses the original name with slashes
3. This is independent of the Triton deployment

## Examples

### HF Method (No Change Needed)
```bash
python measure.py \
    --method hf \
    --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --engine vllm \
    --endpoint http://localhost:8000

# Sends: -m Qwen/Qwen3-30B-A3B-Thinking-2507 ✅
# Works: HF vLLM expects original name
```

### Triton Method (Auto-Fixed)
```bash
python measure.py \
    --method triton \
    --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --engine vllm \
    --endpoint http://localhost:9000

# Sends: -m Qwen_Qwen3-30B-A3B-Thinking-2507 ✅
# Works: Matches Triton model repository name
```

## Triton Model Repository Structure

```
.cache/triton/model_repository/
└── Qwen_Qwen3-30B-A3B-Thinking-2507/  ← Underscore, not slash
    ├── config.pbtxt
    └── 1/
        └── model.json (for vLLM backend)
```

## Container Naming Consistency

Note: Container names also use sanitized format:

```bash
# Container name (from docker.py)
qwen3-30b-a3b-thinking-2507-triton-vllm
# ↑ All lowercase, hyphens, no slashes or underscores

# Triton model name (in repository)
Qwen_Qwen3-30B-A3B-Thinking-2507
# ↑ Original case, underscores replace slashes

# API request model name (after fix)
Qwen_Qwen3-30B-A3B-Thinking-2507
# ↑ Matches repository name
```

## Testing

### Verify Model Name in Triton

```bash
# Check Triton model repository
ls .cache/triton/model_repository/
# Should show: Qwen_Qwen3-30B-A3B-Thinking-2507/

# Check if model is loaded
curl http://localhost:9000/v1/models
# Should return: Qwen_Qwen3-30B-A3B-Thinking-2507
```

### Test Benchmark

```bash
# Should work now
./run_one.sh triton vllm

# Internally calls measure.py with corrected model name
```

## Related Code

### In `docker.py` (TritonModelDeployer)

```python
def _setup_model_repository(self):
    model_name = self.model.replace('/', '_')  # Sanitize for filesystem
    model_dir = model_repo_dir / model_name
    # Creates: .cache/triton/model_repository/Qwen_Qwen3-30B-A3B-Thinking-2507/
```

### In `measure.py` (PerformanceMeasure)

```python
def _build_genai_perf_command(self, mode):
    model_name_for_request = self.model
    if self.method == "triton":
        model_name_for_request = self.model.replace('/', '_')  # Match Triton name
    cmd_parts.append(f"-m {model_name_for_request}")
```

## Summary

✅ **Fixed**: Model name now matches Triton repository naming convention
✅ **Automatic**: Detects `method == "triton"` and sanitizes automatically  
✅ **Backward Compatible**: HF method unchanged (still uses original name)
✅ **Consistent**: Aligns with how docker.py creates model repositories
✅ **Flexible**: Can be extended for other methods (NIM, UNIM) if needed

## Future Methods

When implementing NIM/UNIM, consider their naming conventions:

```python
def _build_genai_perf_command(self, mode):
    model_name_for_request = self.model
    
    if self.method == "triton":
        model_name_for_request = self.model.replace('/', '_')
    elif self.method == "nim":
        # NIM might have different convention
        model_name_for_request = sanitize_for_nim(self.model)
    elif self.method == "unim":
        # UNIM might have different convention
        model_name_for_request = sanitize_for_unim(self.model)
    
    cmd_parts.append(f"-m {model_name_for_request}")
```

