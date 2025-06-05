from plots.plot_functions import plot_griewank, plot_ackley
from experiments.run_experiments import run_experiments

def main():
    print("Welcome to the Optimization Playground!")
    print("What would you like to do?")
    print("1. Create some cool function visualizations")
    print("2. Let's run some optimization experiments")
    choice = input("Pick your adventure (1/2): ")
    if choice == '1':
        print("Time to make some art! Drawing those functions...")
        plot_griewank()
        plot_ackley()
        print("Your masterpieces are waiting in the 'plots/' folder!")
    elif choice == '2':
        print("Let's see what our genetic algorithms can do...")
        run_experiments()
    else:
        print("Oops! That's not a valid option. Try 1 or 2!")

if __name__ == '__main__':
    main()
