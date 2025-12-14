# GenAI-Perf Measurement Mode Fix

## Issue
GenAI-Perf doesn't allow both `--measurement-interval` and `--request-count` arguments at the same time:
```
error: argument --measurement-interval/-p: not allowed with argument --request-count/--num-requests
```

## Solution
Changed the implementation to use **mutually exclusive measurement modes**:

### Mode 1: Count-based (Default)
Use `--request-count` to send a specific number of requests:
```bash
python measure.py \
    --request-count 1000 \
    --concurrency 40 \
    ...
```

### Mode 2: Time-based (Optional)
Use `--measurement-interval` to measure for a specific duration:
```bash
python measure.py \
    --measurement-interval 10000 \  # 10 seconds
    --concurrency 40 \
    ...
```

## Changes Made

### 1. `measure.py`
- Changed default `DEFAULT_MEASUREMENT_INTERVAL` from `10000` to `None`
- Updated `_build_genai_perf_command()` to use only one mode:
  ```python
  if self.request_count:
      cmd_parts.append(f"--request-count {self.request_count}")
  else:
      cmd_parts.append(f"--measurement-interval {self.measurement_interval}")
  ```
- Updated argument help text to clarify mutual exclusivity

### 2. `run_one.sh`
- Removed `MEASUREMENT_INTERVAL` variable
- Removed `--measurement-interval` from `MEASURE_ARGS`
- Now uses count-based mode by default

### 3. `run_all.sh`
- Removed `MEASUREMENT_INTERVAL` variable
- Removed `--measurement-interval` from `MEASURE_ARGS`
- Now uses count-based mode by default

### 4. Documentation
- Updated `README.md` to show both modes
- Updated `QUICK_REFERENCE.md` to explain the difference
- Added note about mutual exclusivity

## Usage

### Default (Count-based)
```bash
./run_one.sh  # Uses --request-count 1000
```

### Custom Count
```bash
python measure.py \
    --request-count 500 \
    ...
```

### Time-based
```bash
python measure.py \
    --measurement-interval 30000 \  # 30 seconds
    # Don't specify --request-count
    ...
```

## Rationale

**Count-based mode (default)** is better for:
- ✅ Reproducible results (always same number of requests)
- ✅ Fair comparisons between methods/engines
- ✅ Predictable test duration (roughly)
- ✅ Statistical significance (known sample size)

**Time-based mode** is better for:
- ⏱️ Fixed duration tests
- 🔄 Continuous monitoring scenarios
- 📊 Time-series data collection

For benchmarking and comparison purposes, **count-based mode is recommended** (and is now the default).

## Verification

All files compile correctly:
```bash
python3 -m py_compile measure.py  # ✅ Success
```

Scripts are executable:
```bash
chmod +x run_one.sh run_all.sh  # ✅ Ready
```

