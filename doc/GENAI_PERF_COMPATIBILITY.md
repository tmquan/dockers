# GenAI-Perf Compatibility Fixes

## Issues Fixed

### 1. Removed `--service-kind openai` argument
**Error:** `genai-perf: error: unrecognized arguments: --service-kind openai`

**Cause:** The `--service-kind` argument is not available in this version of genai-perf.

**Solution:** Removed the `--service-kind openai` argument from the command. GenAI-Perf infers the service type from the endpoint-type and URL automatically.

### 2. Removed `--measurement-interval` with `--request-count`
**Error:** `error: argument --measurement-interval/-p: not allowed with argument --request-count/--num-requests`

**Cause:** GenAI-Perf doesn't allow both arguments simultaneously.

**Solution:** Made them mutually exclusive - uses `--request-count` by default, only uses `--measurement-interval` if explicitly specified without request-count.

## Changes Made

### `measure.py`

1. **Removed `--service-kind openai`** from genai-perf command
   ```python
   # Old:
   "--service-kind openai",
   
   # New:
   # Removed - genai-perf infers from endpoint-type
   ```

2. **Made measurement modes mutually exclusive**
   ```python
   if self.request_count:
       cmd_parts.append(f"--request-count {self.request_count}")
   else:
       cmd_parts.append(f"--measurement-interval {self.measurement_interval}")
   ```

3. **Removed "service_kind" from CSV output**
   - Removed from metrics dict
   - Removed from CSV column list
   - Simplified output format

4. **Changed default measurement mode**
   ```python
   DEFAULT_MEASUREMENT_INTERVAL = None  # Use request-count by default
   ```

### `run_one.sh` and `run_all.sh`

1. **Removed MEASUREMENT_INTERVAL variable**
2. **Removed `--measurement-interval` from MEASURE_ARGS**
3. **Now uses count-based mode by default**

## Current Command Format

```bash
genai-perf profile \
    --random-seed 42 \
    --warmup-request-count 100 \
    -m MODEL \
    --endpoint-type chat \
    -u http://localhost:8000 \
    --concurrency 40 \
    --request-count 1000 \
    --artifact-dir artifacts/MODEL_METHOD_ENGINE \
    --tokenizer MODEL \
    --synthetic-input-tokens-mean 30000 \
    --output-tokens-mean 200 \
    --streaming
```

## Why These Changes Work

1. **Endpoint Type Detection**: GenAI-Perf automatically detects OpenAI-compatible endpoints from:
   - The `--endpoint-type chat` (or `completions`) flag
   - The URL pattern (e.g., `/v1/chat/completions`)
   - No need for explicit `--service-kind`

2. **Measurement Modes**: Two mutually exclusive modes:
   - **Count-based** (default): `--request-count N` - Send N requests
   - **Time-based** (optional): `--measurement-interval MS` - Measure for MS milliseconds

3. **Simpler Output**: Removed redundant "service_kind" column since all our tests use OpenAI-compatible endpoints

## Compatibility

✅ Works with genai-perf versions that:
- Support `--endpoint-type` flag
- Support OpenAI-compatible endpoints
- Require explicit choice between count-based and time-based measurement

## Usage

### Default (Count-based)
```bash
./run_one.sh
# Uses --request-count 1000
```

### Custom Count
```bash
python measure.py \
    --request-count 500 \
    --concurrency 40 \
    ...
```

### Time-based (Alternative)
```bash
python measure.py \
    --measurement-interval 30000 \
    --concurrency 40 \
    # Don't specify --request-count
    ...
```

## Testing

```bash
# Verify syntax
python3 -m py_compile measure.py  # ✅ Pass

# Test command generation
python measure.py \
    --method hf \
    --model test/model \
    --engine vllm \
    --endpoint http://localhost:8000 \
    --input-tokens-mean 100 \
    --output-tokens-mean 10 \
    --request-count 10
```

## Benefits

1. ✅ **Compatible** with current genai-perf version
2. ✅ **Simpler** command structure
3. ✅ **Cleaner** output (removed redundant column)
4. ✅ **Flexible** measurement modes (count or time)
5. ✅ **Automatic** service type detection

