# 🚀 Benchmark Optimization Functions Using Genetic Algorithms

## 📋 Project Documentation

Hey there! This is my project where I explored how Genetic Algorithms (GAs) perform on some classic optimization problems. I focused on the Griewank and Ackley functions, which are pretty interesting test cases for optimization algorithms. I tried different approaches using both binary and real-valued representations, along with various crossover methods to see what works best.

### 🧩 What's Inside?

- My implementations of the Griewank and Ackley functions
- Two different GA approaches: binary and real-valued
- A bunch of crossover methods to play with
- Some cool visualization tools I built
- A framework for running experiments
- Tools for analyzing the results

## ⚙️ How I Structured This

Here's how I organized everything:

```
├── functions/           # Where I put the benchmark functions
│   ├── ackley.py       # My Ackley function implementation
│   └── griewank.py     # My Griewank function implementation
├── ga/                 # The genetic algorithm stuff
│   ├── binary_ga.py    # Binary version of the GA
│   ├── real_ga.py      # Real-valued version
│   └── crossover.py    # Different ways to mix solutions
├── experiments/        # Where I run and store experiments
├── analysis/          # Tools I made for analyzing results
├── plots/             # Where I keep all the visualizations
├── main.py            # The main script to run everything
└── requirements.txt   # What you need to install
```

### What You'll Need

I used:

- Python 3.x
- NumPy (for all the math stuff)
- Matplotlib (for making pretty plots)
- SciPy (for some extra math functions)

## The Fun Stuff: Implementation Details

### The Benchmark Functions

#### Griewank Function

I implemented this one first. It's a pretty tricky function:

- The math looks like this: f(x) = 1 + Σ(x_i²/4000) - Π(cos(x_i/√i))
- It's got lots of local minima (that's what makes it interesting!)
- The best solution is at f(0,...,0) = 0
- You can find it in `functions/griewank.py`

#### Ackley Function

This one's my favorite - it's like a flat landscape with a deep hole in the middle:

- Here's the math: f(x) = -a*exp(-b*√(1/d*Σ(x_i²))) - exp(1/d*Σ(cos(c\*x_i))) + a + exp(1)
- It's mostly flat but has this cool central peak/valley
- The best solution is also at f(0,...,0) = 0
- Check it out in `functions/ackley.py`

### My Genetic Algorithm Implementations

#### Binary GA

I started with this one because it's more traditional:

- Uses binary strings to represent solutions
- Flips bits for mutation
- Has different ways to combine solutions
- You can tweak the population size
- Uses tournament selection (I found this works best)
- Adjustable mutation rate

#### Real-valued GA

This one's more modern and often works better:

- Uses actual numbers instead of binary
- Uses Gaussian mutation (more natural for real numbers)
- Has arithmetic crossover
- Same tournament selection
- Also adjustable population and mutation rates

#### The Crossover Methods I Implemented

I tried several ways to combine solutions:

- Single-point (the classic)
- Two-point (more flexible)
- Uniform (more random)
- Arithmetic (for real numbers)
- Blend (another real-number approach)

## 💻 How to Use This

### Getting Started

1. First, clone this repo
2. Set up a virtual environment (trust me, it's worth it):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install what you need:
   ```bash
   pip install -r requirements.txt
   ```

### Running Things

1. Just run the main script:
   ```bash
   python main.py
   ```
2. You'll get two options:
   - Make some plots (they look pretty cool)
   - Run the optimization experiments

### Tweaking Things

Feel free to play around with:

- The parameters in `experiments/run_experiments.py`
- GA settings in the respective files
- How the plots look in `plots/plot_functions.py`

## 🏁 What You'll Get

### Outputs

- Cool plots in the `plots/` folder
- Experiment results in `experiments/results/`
- Analysis stuff in `analysis/`

### Analysis Tools I Built

- Ways to measure how well things work
- Tools to check if results are meaningful
- Ways to see how quickly solutions converge
- Methods to check solution quality

### Visualizations

I made several types of plots:

- 2D and 3D views of the functions
- Plots showing how solutions improve over time
- Charts showing where solutions end up
- Comparisons of different methods
