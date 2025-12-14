# Input Token Length Fix

## Issue

**Error:**
```
ValueError: This model's maximum context length is 30000 tokens. 
However, your request has 30010 input tokens. 
Please reduce the length of the input messages.
```

## Root Cause

When using `--synthetic-input-tokens-mean`, genai-perf generates synthetic prompts with some **natural variation** around the mean. Even though we specified 30000 tokens with `--synthetic-input-tokens-stddev 0`, the actual generated prompts can slightly exceed the mean due to:

1. **Tokenization differences** between genai-perf's tokenizer and the model's tokenizer
2. **Rounding in token generation**
3. **Chat template overhead** (system messages, special tokens)
4. **Natural variation** in synthetic data generation

## Solution

Reduced the input token mean from **30000 to 28000** tokens, leaving approximately **~10% buffer** (2000 tokens) to accommodate:
- Tokenization variations
- Chat template overhead
- Generation variance
- Safety margin

## Changes Made

### 1. `run_one.sh`
```bash
# Old
ACTUAL_INPUT_LEN=30000        # No buffer

# New  
ACTUAL_INPUT_LEN=28000        # Leave ~10% buffer for variation
INPUT_SEQUENCE_LENGTH=32768   # Max model length (deployment setting)
```

### 2. `run_all.sh`
```bash
# Old
ACTUAL_INPUT_LEN=30000        # No buffer

# New
ACTUAL_INPUT_LEN=28000        # Leave ~10% buffer for variation  
INPUT_SEQUENCE_LENGTH=32768   # Max model length (deployment setting)
```

## Key Distinction

- **`INPUT_SEQUENCE_LENGTH`** (32768): Maximum context length configured for the **model deployment**
  - Used in `docker.py start --max-model-len 32768`
  - This is the absolute maximum the model can handle

- **`ACTUAL_INPUT_LEN`** (28000): Mean for **synthetic input generation** in benchmarks
  - Used in `measure.py --input-tokens-mean 28000`
  - This should be **less than** the model's max to leave buffer

## Recommended Buffers

| Model Max Context | Recommended Input Mean | Buffer |
|-------------------|------------------------|--------|
| 8,192 | 7,000 | ~15% |
| 16,384 | 14,000 | ~15% |
| 30,000 | 28,000 | ~7% |
| 32,768 | 30,000 | ~8% |
| 65,536 | 60,000 | ~8% |
| 131,072 | 120,000 | ~8% |

## Why Not Exact Match?

You might think "just set it to 29,999" - but here's why that's risky:

1. **Tokenization Variance**: Different tokenizers may count tokens differently
2. **Chat Templates**: System messages add tokens automatically
3. **Special Tokens**: `<|im_start|>`, `<|im_end|>`, etc. add overhead
4. **Safety**: Better to leave margin than hit errors repeatedly

## Best Practices

### For Benchmarking
```bash
# Safe: Leave 5-10% buffer
python measure.py --input-tokens-mean 28000  # ✅ Safe

# Risky: Too close to limit
python measure.py --input-tokens-mean 29900  # ⚠️ May fail

# Unsafe: At or above limit
python measure.py --input-tokens-mean 30000  # ❌ Will fail
```

### For Model Deployment
```bash
# Deploy with full context capability
python docker.py start \
    --model MODEL \
    --engine vllm \
    --max-model-len 32768  # Full capability

# Benchmark with safe buffer
python measure.py \
    --input-tokens-mean 28000 \  # Safe mean
    --output-tokens-mean 200      # Fits comfortably
```

## Testing Different Context Lengths

```bash
# Short context (8k)
./run_one.sh hf vllm MODEL 8000
# Internally uses safe input like 7000 tokens

# Medium context (16k)  
./run_one.sh hf vllm MODEL 8000
# Internally uses safe input like 14000 tokens

# Long context (32k) - default
./run_one.sh
# Uses 28000 token input (safe for 30k-32k models)
```

## Verification

After this fix:
```bash
# Should work without errors
./run_one.sh

# Tokens generated will be ~28000 ±variance
# Well within 30000-32768 limit
```

## Related Configurations

All these work together:
1. **Model deployment**: `--max-model-len 32768` (maximum capability)
2. **Benchmark input**: `--input-tokens-mean 28000` (safe mean)
3. **Output tokens**: `--output-tokens-mean 200` (fits easily)
4. **Total**: ~28200 tokens << 30000 limit ✅

## Summary

✅ **Fixed**: Reduced input mean from 30000 → 28000 tokens
✅ **Buffer**: ~2000 token safety margin (~7%)
✅ **Safe**: Accommodates tokenization variance and overhead
✅ **Flexible**: Works with 30k-32k context models

