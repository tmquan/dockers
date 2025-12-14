# Documentation Index

Welcome to the Unified LLM Deployment and Benchmarking Suite documentation!

**Location:** All documentation is in the `doc/` folder, except `README.md` which stays in the root.

## 📚 Documentation Files

### Getting Started
1. **[QUICKSTART.md](QUICKSTART.md)** ⭐ **START HERE**
   - Get running in 5 minutes
   - Prerequisites and setup
   - Basic examples
   - Common troubleshooting

2. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** 📖
   - Command cheat sheet
   - Supported combinations
   - Common issues and fixes
   - Quick examples

### Complete Documentation
3. **[README.md](../README.md)** 📘
   - Full system documentation
   - Detailed usage examples
   - Configuration options
   - Performance tips
   - Contributing guidelines

### Architecture and Design
4. **[ARCHITECTURE.md](ARCHITECTURE.md)** 🏗️
   - System design diagrams
   - Class hierarchy
   - Data flow
   - Extension points
   - Design principles

5. **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)** 📝
   - What changed and why
   - Migration guide
   - Breaking changes
   - Benefits delivered

6. **[IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)** ✅
   - Implementation summary
   - Success criteria
   - Testing checklist
   - Next steps

## 🚀 Core Scripts

### Deployment
- **`docker.py`** - Unified deployment manager
  ```bash
  python docker.py start --method hf --model MODEL --engine vllm
  ```

- **`measure.py`** - Performance measurement
  ```bash
  python measure.py --method hf --model MODEL --engine vllm --endpoint URL
  ```

### Testing
- **`run_one.sh`** - Quick single test
  ```bash
  ./run_one.sh [METHOD] [ENGINE] [MODEL] [PORT]
  ```

- **`run_all.sh`** - Full benchmark suite
  ```bash
  ./run_all.sh
  ```

All scripts are located in the parent directory (`../`).

## 📊 Quick Navigation

### I want to...

#### ...get started quickly
→ [QUICKSTART.md](QUICKSTART.md) - 5-minute guide

#### ...run a single test
→ [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - See "Single Test" section

#### ...understand the architecture
→ [ARCHITECTURE.md](ARCHITECTURE.md) - Visual diagrams and design

#### ...learn all the commands
→ [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Complete command reference

#### ...see detailed examples
→ [README.md](README.md) - Comprehensive usage guide

#### ...add a new method/engine
→ [ARCHITECTURE.md](ARCHITECTURE.md) - See "Extension Points" section

#### ...understand what changed
→ [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) - Complete change log

#### ...troubleshoot an issue
→ [QUICKSTART.md](QUICKSTART.md) - See "Troubleshooting" section
→ [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - See "Common Issues" section

## 🎯 Use Case Guides

### Research / Experimentation
1. Start with [QUICKSTART.md](QUICKSTART.md)
2. Use `./run_one.sh` for quick tests
3. Read [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for variations

### Production Deployment
1. Read [README.md](README.md) completely
2. Review [ARCHITECTURE.md](ARCHITECTURE.md) for design
3. Test with `./run_one.sh`
4. Deploy with `docker.py` directly

### Performance Benchmarking
1. Use `./run_all.sh` for comprehensive testing
2. Analyze results with pandas (examples in README.md)
3. Tune parameters based on [README.md](README.md) Performance Tips

### Development / Extension
1. Understand design in [ARCHITECTURE.md](ARCHITECTURE.md)
2. Review base class in `docker.py`
3. Follow extension patterns in [ARCHITECTURE.md](ARCHITECTURE.md)
4. Test with existing scripts

## 📖 Reading Order

### For First-Time Users
1. [QUICKSTART.md](QUICKSTART.md) - Get running
2. [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Learn commands
3. [README.md](../README.md) - Deep dive when needed

### For Developers
1. [ARCHITECTURE.md](ARCHITECTURE.md) - Understand design
2. [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) - See what changed
3. [README.md](../README.md) - Reference implementation

### For DevOps/SRE
1. [QUICKSTART.md](QUICKSTART.md) - Quick validation
2. [README.md](../README.md) - Configuration and tuning
3. [ARCHITECTURE.md](ARCHITECTURE.md) - System understanding

## 🔍 Finding Information

### By Topic

#### Configuration
- [README.md](../README.md) - "Configuration" section
- `../docker.py` - Top of file (DEFAULT_* variables)
- `../measure.py` - Top of file (DEFAULT_* variables)

#### Commands
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - All commands
- [QUICKSTART.md](QUICKSTART.md) - Common operations

#### Examples
- [README.md](../README.md) - "Usage Examples" section
- [QUICKSTART.md](QUICKSTART.md) - Quick examples
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Command examples

#### Troubleshooting
- [QUICKSTART.md](QUICKSTART.md) - "Troubleshooting" section
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - "Common Issues" section

#### Architecture
- [ARCHITECTURE.md](ARCHITECTURE.md) - Complete design
- [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) - Implementation details

## 📦 File Organization

```
dockers/
├── README.md ⭐ Main documentation (root level)
│
├── doc/                          Documentation folder
│   ├── INDEX.md                  This file - documentation index
│   ├── QUICKSTART.md             5-minute getting started
│   ├── QUICK_REFERENCE.md        Command cheat sheet
│   ├── ARCHITECTURE.md           System design diagrams
│   ├── REFACTORING_SUMMARY.md    Change log and migration
│   ├── IMPLEMENTATION_COMPLETE.md Status and checklist
│   └── GENAI_PERF_FIX.md         GenAI-Perf fix notes
│
├── Core Scripts
│   ├── docker.py
│   ├── measure.py
│   ├── run_one.sh
│   └── run_all.sh
│
├── Legacy (Reference only)
│   ├── docker_hf.py
│   ├── docker_hf_with_triton.py
│   ├── run_benchmark.sh
│   ├── run_benchmark_with_triton.sh
│   └── measure_perf.py
│
└── Generated (Auto-created)
    ├── .cache/
    └── artifacts/
```

## 🔗 Quick Links

### Most Used
- [QUICKSTART.md](QUICKSTART.md) - Start here
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Command reference
- [README.md](README.md) - Complete docs

### Reference
- [ARCHITECTURE.md](ARCHITECTURE.md) - Design docs
- [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) - Change log
- [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md) - Status

## 💡 Tips

1. **First time?** Start with [QUICKSTART.md](QUICKSTART.md)
2. **Need a command?** Check [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
3. **Deep dive?** Read [README.md](README.md)
4. **Extending?** See [ARCHITECTURE.md](ARCHITECTURE.md)
5. **Issues?** Check troubleshooting sections

## 📞 Getting Help

1. Check the relevant documentation file above
2. Review example commands in [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
3. Look at troubleshooting sections
4. Check container logs: `docker logs CONTAINER_NAME`
5. Review the code comments in source files

## ✅ Verification

All documentation is:
- ✅ Complete and up-to-date
- ✅ Cross-referenced
- ✅ Example-rich
- ✅ Well-organized
- ✅ Ready to use

Happy deploying! 🚀

