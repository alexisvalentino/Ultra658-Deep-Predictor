# Ultra658 Deep Predictor

A deep learning-based predictor using TensorFlow with temperature scaling and parallel feature processing for intelligent number generation.

## Model Architecture

### 1. Feature Engineering
- Short and long-term window analysis (10 and 50 draws)
- Decayed frequency counting (decay rate: 0.96)
- Time since last appearance tracking
- Momentum indicators (short vs long-term rates)
- Parallel processing with ThreadPoolExecutor

### 2. Neural Network Structure
```python
model = Sequential([
    Dense(128, activation='relu'),      # Input layer
    BatchNormalization(),               # Normalize activations
    Dropout(0.3),                       # Prevent overfitting
    Dense(64, activation='relu'),       # Hidden layer
    BatchNormalization(),               # Normalize activations
    Dropout(0.3),                       # Additional regularization
    Dense(1, activation='sigmoid')      # Output probability
])
```

### 3. Probability Calibration
- Temperature scaling for probability calibration
- Grid search for optimal temperature parameter
- Validation-based calibration fitting

## Installation

1. Create virtual environment (Python 3.10 recommended):
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies:
```powershell
pip install pandas numpy tensorflow scikit-learn ttkthemes
```

## Model Performance Features

### Processing Optimizations
- Multi-threaded feature computation
- Model caching in models/model.h5
- Asynchronous GUI updates
- Progress indication during generation

### Sampling Constraints
- Even number count: 2-4
- Minimum spread: 10
- Maximum sampling attempts: 1000
- Normalized probability sampling

### Output Features
- Combination generation with confidence scores
- Historical tracking (last 5 combinations)
- File export with timestamps
- Clipboard integration

## Technical Parameters
- Short window size: 10 draws
- Long window size: 50 draws
- Decay factor: 0.96
- Training epochs: 100 with early stopping
- Batch size: 512
- Validation split: 15%
- Binary cross-entropy loss
- Adam optimizer
- Dropout rate: 0.3

## File Structure
```
Predictive model/
├── lstmodel.py          # Main application
├── combinations.csv     # Historical data
├── models/             # Model cache
│   └── model.h5
├── output/             # Generated files
│   ├── combinations_*.txt
│   └── history_*.txt
└── predictor.log       # Application logs
```

## Contributing
Contributions welcome! Please submit Pull Requests.

## License
MIT License

## Author
Alexis Valentino

## Disclaimer
This project is for educational purposes only. The model analyzes historical patterns but cannot predict future lottery outcomes. No guarantee of winning is implied or promised.