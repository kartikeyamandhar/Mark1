# Mark1 🚀 - Mac M4 Air ML Stress Test

> **Why "Mark1"?** This is my very first project after making the big switch from being a lifelong Windows user to macOS. Consider this my "Mark 1" prototype as I explore the incredible world of Mac development and machine learning on Apple Silicon.

---

## What is Mark1?

Mark1 is a comprehensive machine learning stress test specifically designed to push the M4 Mac Air to its absolute limits. After years of Windows development, I wanted to see what this little powerhouse could really do.


## 🎯 What This Test Does

Mark1 puts your M4 Air through six intense challenges:

1. **Massive Data Generation** - Creates and processes huge datasets (1M samples, 3000+ features)
2. **Parallel ML Training** - Trains multiple models simultaneously (RandomForest, XGBoost, LightGBM, etc.)
3. **TensorFlow + Metal GPU** - Deep learning with Apple's Metal Performance Shaders
4. **PyTorch + MPS** - Neural networks accelerated by Metal Performance Shaders
5. **Hyperparameter Tuning** - Grid search across thousands of parameter combinations
6. **CPU Saturation** - Multiprocessing ensemble that uses every core

## 🔧 Dependencies & Installation

### System Requirements
- Mac with M4 chip (obviously!)
- macOS Sonoma or later
- At least 16GB RAM recommended
- Python 3.9+

### Installing Dependencies

```bash
# First, install Homebrew if you haven't (coming from Windows, this is like Chocolatey)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python if needed
brew install python

# Install required packages
pip install numpy pandas matplotlib seaborn scikit-learn
pip install xgboost lightgbm psutil
pip install tensorflow-metal  # This is the magic for M4 GPU acceleration
pip install torch torchvision  # PyTorch with MPS support
```

### Quick One-Liner Installation
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm psutil tensorflow-metal torch torchvision
```

## 🚀 Running the Test

```bash
# Clone or download the repository
cd path/to/Mark1

# Run the stress test
python main.py
```

**Pro tip from a Windows convert:** Open Activity Monitor while running this. Watch those CPU cores light up!

## 📊 What to Expect

Here's what you'll see:

- Real-time progress updates for each test
- System resource monitoring
- Performance scores for each model
- Final performance grade (🚀 EXCELLENT to ⏳ AVERAGE)
- Beautiful visualization saved as PNG

## 🎮 Monitoring Your System

While Mark1 runs, I highly recommend watching Activity Monitor to see:
- **CPU**: All cores maxed out during parallel processing
- **GPU**: Metal acceleration in action
- **Memory**: How efficiently the M4 handles large datasets
- **Temperature**: Thermal management (spoiler: it's impressive)

## 📈 Results Interpretation

Mark1 generates a comprehensive report showing:
- Execution time for each test
- Model accuracy scores
- Memory usage statistics
- Overall performance grade

## Why I Built This

As someone who spent years on Windows doing ML work, I wanted to really understand what I was getting with the M4 Air. The benchmarks online are great, but nothing beats pushing your own code through the system.



## 🐛 Troubleshooting

### Common Issues I Encountered:

**TensorFlow Metal Issues:**
```bash
# If you get Metal errors, try:
pip install --upgrade tensorflow-metal
```

**PyTorch MPS Problems:**
```bash
# Ensure you have the latest PyTorch:
pip install --upgrade torch torchvision
```

**Permission Errors (familiar from Windows!):**
```bash
# Use Python user installs:
pip install --user [package_name]
```

## 🛣️ What's Next?

This is just Mark1. I'm already planning Mark2 with:
- Computer vision benchmarks
- NLP model training
- Real-world dataset processing
- iOS integration experiments

## 🤝 Contributing

Found a bug? Have ideas for Mark2? I'm all ears! This is a learning project, and I'd love input from other developers, especially those who've made similar transitions.

