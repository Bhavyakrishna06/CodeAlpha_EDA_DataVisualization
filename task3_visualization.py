import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_task3_visualization():
    print("=" * 50)
    print("TASK 3: DATA VISUALIZATION")
    print("=" * 50)

    filepath = "data/cleaned_ecommerce.csv"
    
    # 1. Check if Task 2 data exists
    if not os.path.exists(filepath):
        print(f"[!] Error: '{filepath}' not found.")
        print("[!] Please run 'task2.py' first to generate the dataset!")
        return

    # 2. Load the data
    df = pd.read_csv(filepath)
    print("[+] Cleaned dataset loaded successfully.")

    # Set visualization theme
    sns.set_theme(style="whitegrid")
    
    # Create a 2x2 grid for multiple charts
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('E-Commerce Sales Insights Dashboard', fontsize=18, fontweight='bold')

    # Chart 1: Average Spend by Category (Bar Chart)
    sns.barplot(ax=axes[0, 0], data=df, x='Category', y='Total_Amount', errorbar=None, palette='viridis')
    axes[0, 0].set_title('Average Spend per Category', fontsize=14)
    axes[0, 0].set_ylabel('Average Amount ($)')

    # Chart 2: Customer Age Distribution (Histogram)
    sns.histplot(ax=axes[0, 1], data=df, x='Customer_Age', bins=20, kde=True, color='royalblue')
    axes[0, 1].set_title('Customer Age Distribution', fontsize=14)
    axes[0, 1].set_xlabel('Age')

    # Chart 3: Spend Distribution by Gender (Box Plot)
    sns.boxplot(ax=axes[1, 0], data=df, x='Gender', y='Total_Amount', palette='pastel')
    axes[1, 0].set_title('Total Spend Distribution by Gender', fontsize=14)
    axes[1, 0].set_ylabel('Total Amount ($)')

    # Chart 4: Total Transactions per Category (Count Plot)
    sns.countplot(ax=axes[1, 1], data=df, x='Category', palette='Set2')
    axes[1, 1].set_title('Number of Transactions per Category', fontsize=14)
    axes[1, 1].set_ylabel('Transaction Count')

    # Adjust layout so charts don't overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the dashboard as an image
    output_image = "data/dashboard_visuals.png"
    plt.savefig(output_image, dpi=300)
    print(f"\n[+] Success! Visualization dashboard saved as: '{output_image}'")
    
    # Display the charts on your screen
    plt.show()

if __name__ == "__main__":
    run_task3_visualization()
