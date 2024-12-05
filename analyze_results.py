import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

def analyze_poison_probability_results():
    # Your experimental results
    results = {
        0.1: {'mean': 3.2472, 'std': 0.6060},
        0.2: {'mean': 3.1725, 'std': 0.6367},
        0.3: {'mean': 3.2019, 'std': 0.5085},
        0.4: {'mean': 2.9791, 'std': 0.4917},
        0.5: {'mean': 3.1356, 'std': 0.5786}
    }

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame({
        'probability': list(results.keys()),
        'mean_reward': [v['mean'] for v in results.values()],
        'std_reward': [v['std'] for v in results.values()]
    })

    # Create figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Mean Rewards with Error Bars
    ax1.errorbar(df['probability'], df['mean_reward'], 
                yerr=df['std_reward'], fmt='o-', capsize=5)
    ax1.set_xlabel('Poison Probability')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Agent Performance vs. Poison Probability')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Distribution of Rewards
    sns.violinplot(data=df, x='probability', y='mean_reward', ax=ax2)
    ax2.set_xlabel('Poison Probability')
    ax2.set_ylabel('Reward Distribution')
    ax2.set_title('Reward Distribution by Poison Probability')

    plt.tight_layout()
    plt.savefig('results/poison_probability_analysis.png')
    plt.close()

    # Statistical Analysis
    print("\nStatistical Analysis:")
    print("-" * 50)
    
    # Calculate correlation
    correlation = stats.pearsonr(df['probability'], df['mean_reward'])
    print(f"Correlation coefficient: {correlation[0]:.4f}")
    print(f"P-value: {correlation[1]:.4f}")

    # Summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    print(f"Overall mean reward: {df['mean_reward'].mean():.4f}")
    print(f"Overall std deviation: {df['mean_reward'].std():.4f}")
    print(f"Best performing probability: {df.loc[df['mean_reward'].idxmax(), 'probability']}")
    print(f"Worst performing probability: {df.loc[df['mean_reward'].idxmin(), 'probability']}")

def plot_learning_curves():
    """
    Create learning curves from the training data
    Assuming we have training data saved in results/training_data.csv
    """
    try:
        # Load training data
        df = pd.read_csv('results/collab_rate_results.csv')
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['probability'], df['average_reward'], 'b-', label='Average Reward')
        plt.fill_between(df['probability'], 
                        df['average_reward'] - df['std_reward'],
                        df['average_reward'] + df['std_reward'],
                        alpha=0.2)
        
        plt.xlabel('Poison Probability')
        plt.ylabel('Average Reward')
        plt.title('Learning Performance Across Different Poison Probabilities')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig('results/learning_curves.png')
        plt.close()
        
    except FileNotFoundError:
        print("Training data file not found. Skipping learning curves plot.")

def main():
    # Create results directory if it doesn't exist
    import os
    os.makedirs('results', exist_ok=True)
    
    # Run analysis
    analyze_poison_probability_results()
    plot_learning_curves()
    
    # Print recommendations
    print("\nRecommendations:")
    print("-" * 50)
    print("1. Optimal poison probability appears to be around 0.1")
    print("2. Performance is relatively stable across different probabilities")
    print("3. Consider testing with more extreme probabilities (0.0-0.1) to find optimal point")
    print("4. Higher probabilities (>0.4) show slightly decreased performance")
    
    # Save results to text file
    with open('results/analysis_summary.txt', 'w') as f:
        f.write("Analysis Summary\n")
        f.write("=" * 50 + "\n")
        f.write("\nProbability vs Average Reward:\n")
        f.write("0.1: 3.2472 ± 0.6060\n")
        f.write("0.2: 3.1725 ± 0.6367\n")
        f.write("0.3: 3.2019 ± 0.5085\n")
        f.write("0.4: 2.9791 ± 0.4917\n")
        f.write("0.5: 3.1356 ± 0.5786\n")
        f.write("\nKey Findings:\n")
        f.write("1. Best performance at 0.1 probability\n")
        f.write("2. Most consistent performance at 0.4 probability\n")
        f.write("3. Overall stable performance across probabilities\n")

if __name__ == "__main__":
    main()