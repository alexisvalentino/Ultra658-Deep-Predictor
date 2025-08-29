# Ultra658 Deep Predictor

A deep learning-based predictor for Ultra Lotto 6/58, combining neural networks with MCMC sampling for intelligent lottery number generation.

## Description
This project uses advanced machine learning techniques to generate lottery combinations (6 unique numbers from 1-58) through neural networks and MCMC sampling. The system learns from historical data to propose statistically informed number combinations.

## Features
- Neural network-based prediction model
- MCMC sampling for combination generation
- User-friendly GUI interface
- Historical data analysis
- Confidence scoring system
- Combination save/export functionality

## Prerequisites
- Python 3.10 (Note: Python 3.13 may cause Tcl/Tk issues)
- Required packages:
  ```
  pandas
  numpy
  tensorflow==2.10.0
  scikit-learn
  ```

## Installation

1. Clone the repository
```bash
git clone https://github.com/alexisvalentino/Ultra658-Deep-Predictor.git
cd Ultra658-Deep-Predictor
```

2. Create and activate virtual environment
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

3. Install dependencies
```bash
pip install pandas numpy tensorflow==2.10.0 scikit-learn
```

4. Create initial dataset
```bash
python csvcreation.py
```

## Usage

### Running the Enhanced Predictor
```bash
python enhanced.py
```

The GUI will display:
- Generated combination
- Confidence score
- History of previous generations
- Options to copy or save combinations

## Technical Implementation

### Neural Network Architecture
- Input layer with feature engineering
- Hidden layers with dropout for regularization
- Softmax output layer for number probabilities
- MCMC sampling for final combination generation

### Probability Analysis

#### 1. Theoretical Probability of Ultra Lotto 6/58
- **Total Combinations**: The number of possible combinations is given by the binomial coefficient:
  ```
  C(58,6) = 58!/(6!(58-6)!) = 40,475,358
  ```
  Each combination (e.g., `05-13-25-33-41-49`) has an equal probability:
  ```
  P(any combination) = 1/40,475,358 ≈ 2.47 × 10^-8
  ```
- **Randomness**: In a fair lottery, each number (1-58) has an equal chance of selection, subject to no repeats.

#### 2. Enhanced Model Randomness
Our model implements:
- **Feature Engineering**: Extracts patterns from 613 historical draws
- **Neural Network**: Multi-layer perceptron with dropout layers:
  ```python
  model = Sequential([
      Dense(64, activation='relu'),
      Dropout(0.2),
      Dense(32, activation='relu'),
      Dropout(0.2),
      Dense(58, activation='softmax')
  ])
  ```
- **MCMC Sampling**: Uses Metropolis-Hastings algorithm with temperature control:
  ```
  P(accept) = min(1, exp((new_score - best_score)/temp))
  ```

#### 3. Probability Distribution
- **Combined Approach**: 
  - 70% Neural network predictions
  - 30% Historical frequency
- **Temperature Parameter**: Controls exploration vs exploitation
- **Effective Coverage**: Dataset covers approximately 0.0015% of possible combinations

### Data Processing
- Historical draw analysis
- Feature extraction
- Probability distribution modeling
- Statistical validation

## Model Limitations

1. **Statistical Constraints**
   - Small dataset (613 draws) limits pattern learning
   - True lottery randomness makes perfect prediction impossible

2. **Predictive Power**
   - Model generates statistically plausible combinations
   - No guarantee of future draw prediction
   - Serves educational and simulation purposes only

## Physics-Inspired Implementation
- MCMC sampling mimics physical lottery tumbler dynamics
- Temperature parameter simulates thermal fluctuations
- Probability distribution reflects energy states of the system

## Troubleshooting

### Tcl/Tk Error
If you encounter `_tkinter.TclError: Can't find a usable init.tcl`, follow these steps:

1. Reinstall Python 3.10
   - Download from [python.org](https://www.python.org/downloads/release/python-31011/)
   - Use "Customize installation" with tcl/tk option

2. Recreate virtual environment
```bash
rmdir /s .venv
python -m venv .venv
.\.venv\Scripts\activate
pip install pandas numpy tensorflow==2.10.0 scikit-learn
```

### Missing Data File
If `combinations.csv` is missing:
```bash
python csvcreation.py
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
MIT License

## Author
Alexis Valentino

## Disclaimer
This project is for educational purposes only. No guarantee of lottery success is implied or promised.