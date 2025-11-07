#!/usr/bin/env python3
"""
Script to plot training results and loss curves
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_sample_training_data():
    """Create sample training data for demonstration"""
    
    # Collaborative Filtering Training Data
    cf_epochs = np.arange(1, 51)
    cf_train_loss = 0.8 * np.exp(-cf_epochs/15) + 0.1 + 0.05 * np.random.normal(0, 1, 50)
    cf_val_loss = 0.9 * np.exp(-cf_epochs/12) + 0.15 + 0.08 * np.random.normal(0, 1, 50)
    cf_rmse = 1.2 * np.exp(-cf_epochs/10) + 0.3 + 0.1 * np.random.normal(0, 1, 50)
    
    # Content-Based Filtering Training Data
    cbf_epochs = np.arange(1, 51)
    cbf_train_loss = 0.7 * np.exp(-cbf_epochs/18) + 0.12 + 0.06 * np.random.normal(0, 1, 50)
    cbf_val_loss = 0.85 * np.exp(-cbf_epochs/14) + 0.18 + 0.09 * np.random.normal(0, 1, 50)
    cbf_accuracy = 0.3 + 0.6 * (1 - np.exp(-cbf_epochs/12)) + 0.05 * np.random.normal(0, 1, 50)
    
    # Hybrid Model Training Data
    hybrid_epochs = np.arange(1, 51)
    hybrid_train_loss = 0.6 * np.exp(-hybrid_epochs/20) + 0.08 + 0.04 * np.random.normal(0, 1, 50)
    hybrid_val_loss = 0.7 * np.exp(-hybrid_epochs/16) + 0.12 + 0.06 * np.random.normal(0, 1, 50)
    hybrid_ndcg = 0.2 + 0.7 * (1 - np.exp(-hybrid_epochs/15)) + 0.03 * np.random.normal(0, 1, 50)
    
    return {
        'cf': {'epochs': cf_epochs, 'train_loss': cf_train_loss, 'val_loss': cf_val_loss, 'rmse': cf_rmse},
        'cbf': {'epochs': cbf_epochs, 'train_loss': cbf_train_loss, 'val_loss': cbf_val_loss, 'accuracy': cbf_accuracy},
        'hybrid': {'epochs': hybrid_epochs, 'train_loss': hybrid_train_loss, 'val_loss': hybrid_val_loss, 'ndcg': hybrid_ndcg}
    }

def plot_collaborative_filtering(data, save_path):
    """Plot Collaborative Filtering training results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss curves
    ax1.plot(data['epochs'], data['train_loss'], 'b-', linewidth=2, label='Training Loss', alpha=0.8)
    ax1.plot(data['epochs'], data['val_loss'], 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
    ax1.fill_between(data['epochs'], data['train_loss'], alpha=0.3, color='blue')
    ax1.fill_between(data['epochs'], data['val_loss'], alpha=0.3, color='red')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Collaborative Filtering - Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RMSE curve
    ax2.plot(data['epochs'], data['rmse'], 'g-', linewidth=2, label='RMSE', alpha=0.8)
    ax2.fill_between(data['epochs'], data['rmse'], alpha=0.3, color='green')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('RMSE')
    ax2.set_title('Collaborative Filtering - RMSE', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Collaborative Filtering plot saved: {save_path}")

def plot_content_based_filtering(data, save_path):
    """Plot Content-Based Filtering training results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss curves
    ax1.plot(data['epochs'], data['train_loss'], 'b-', linewidth=2, label='Training Loss', alpha=0.8)
    ax1.plot(data['epochs'], data['val_loss'], 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
    ax1.fill_between(data['epochs'], data['train_loss'], alpha=0.3, color='blue')
    ax1.fill_between(data['epochs'], data['val_loss'], alpha=0.3, color='red')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Content-Based Filtering - Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(data['epochs'], data['accuracy'], 'purple', linewidth=2, label='Accuracy', alpha=0.8)
    ax2.fill_between(data['epochs'], data['accuracy'], alpha=0.3, color='purple')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Content-Based Filtering - Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Content-Based Filtering plot saved: {save_path}")

def plot_hybrid_model(data, save_path):
    """Plot Hybrid Model training results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss curves
    ax1.plot(data['epochs'], data['train_loss'], 'b-', linewidth=2, label='Training Loss', alpha=0.8)
    ax1.plot(data['epochs'], data['val_loss'], 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
    ax1.fill_between(data['epochs'], data['train_loss'], alpha=0.3, color='blue')
    ax1.fill_between(data['epochs'], data['val_loss'], alpha=0.3, color='red')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Hybrid Model - Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # NDCG curve
    ax2.plot(data['epochs'], data['ndcg'], 'orange', linewidth=2, label='NDCG@10', alpha=0.8)
    ax2.fill_between(data['epochs'], data['ndcg'], alpha=0.3, color='orange')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('NDCG@10')
    ax2.set_title('Hybrid Model - NDCG@10', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Hybrid Model plot saved: {save_path}")

def plot_comparison(data, save_path):
    """Plot comparison of all models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Loss comparison
    ax1.plot(data['cf']['epochs'], data['cf']['val_loss'], 'b-', linewidth=2, label='Collaborative Filtering', alpha=0.8)
    ax1.plot(data['cbf']['epochs'], data['cbf']['val_loss'], 'r-', linewidth=2, label='Content-Based Filtering', alpha=0.8)
    ax1.plot(data['hybrid']['epochs'], data['hybrid']['val_loss'], 'g-', linewidth=2, label='Hybrid Model', alpha=0.8)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Model Comparison - Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Performance metrics comparison
    ax2.plot(data['cf']['epochs'], data['cf']['rmse'], 'b-', linewidth=2, label='CF - RMSE', alpha=0.8)
    ax2.plot(data['cbf']['epochs'], data['cbf']['accuracy'], 'r-', linewidth=2, label='CBF - Accuracy', alpha=0.8)
    ax2.plot(data['hybrid']['epochs'], data['hybrid']['ndcg'], 'g-', linewidth=2, label='Hybrid - NDCG@10', alpha=0.8)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Performance Metrics')
    ax2.set_title('Model Comparison - Performance Metrics', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Model Comparison plot saved: {save_path}")

def plot_training_summary(data, save_path):
    """Plot training summary with key metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Final performance metrics
    models = ['Collaborative\nFiltering', 'Content-Based\nFiltering', 'Hybrid Model']
    final_losses = [data['cf']['val_loss'][-1], data['cbf']['val_loss'][-1], data['hybrid']['val_loss'][-1]]
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    bars1 = ax1.bar(models, final_losses, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Final Validation Loss')
    ax1.set_title('Final Validation Loss Comparison', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars1, final_losses):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Training time (simulated)
    training_times = [45, 38, 52]  # minutes
    bars2 = ax2.bar(models, training_times, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax2.set_ylabel('Training Time (minutes)')
    ax2.set_title('Training Time Comparison', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars2, training_times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value} min', ha='center', va='bottom', fontweight='bold')
    
    # Convergence speed
    convergence_epochs = [25, 30, 20]  # epochs to converge
    bars3 = ax3.bar(models, convergence_epochs, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax3.set_ylabel('Epochs to Converge')
    ax3.set_title('Convergence Speed', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars3, convergence_epochs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{value} epochs', ha='center', va='bottom', fontweight='bold')
    
    # Model complexity (parameters)
    parameters = [125000, 98000, 180000]  # number of parameters
    bars4 = ax4.bar(models, parameters, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax4.set_ylabel('Number of Parameters')
    ax4.set_title('Model Complexity', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars4, parameters):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000, 
                f'{value:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training Summary plot saved: {save_path}")

def main():
    """Main function to generate all training plots"""
    print("Generating Training Plots...")
    
    # Create output directory
    output_dir = "training_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate sample data
    print("Creating sample training data...")
    data = create_sample_training_data()
    
    # Generate plots
    print("Generating plots...")
    
    # Individual model plots
    plot_collaborative_filtering(data['cf'], os.path.join(output_dir, "01_collaborative_filtering.png"))
    plot_content_based_filtering(data['cbf'], os.path.join(output_dir, "02_content_based_filtering.png"))
    plot_hybrid_model(data['hybrid'], os.path.join(output_dir, "03_hybrid_model.png"))
    
    # Comparison plots
    plot_comparison(data, os.path.join(output_dir, "04_model_comparison.png"))
    plot_training_summary(data, os.path.join(output_dir, "05_training_summary.png"))
    
    # Save training data as JSON for reference
    training_info = {
        "timestamp": datetime.now().isoformat(),
        "models": {
            "collaborative_filtering": {
                "final_train_loss": float(data['cf']['train_loss'][-1]),
                "final_val_loss": float(data['cf']['val_loss'][-1]),
                "final_rmse": float(data['cf']['rmse'][-1]),
                "epochs": 50
            },
            "content_based_filtering": {
                "final_train_loss": float(data['cbf']['train_loss'][-1]),
                "final_val_loss": float(data['cbf']['val_loss'][-1]),
                "final_accuracy": float(data['cbf']['accuracy'][-1]),
                "epochs": 50
            },
            "hybrid_model": {
                "final_train_loss": float(data['hybrid']['train_loss'][-1]),
                "final_val_loss": float(data['hybrid']['val_loss'][-1]),
                "final_ndcg": float(data['hybrid']['ndcg'][-1]),
                "epochs": 50
            }
        }
    }
    
    with open(os.path.join(output_dir, "training_results.json"), 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"\nAll plots generated successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Generated files:")
    print(f"   - 01_collaborative_filtering.png")
    print(f"   - 02_content_based_filtering.png") 
    print(f"   - 03_hybrid_model.png")
    print(f"   - 04_model_comparison.png")
    print(f"   - 05_training_summary.png")
    print(f"   - training_results.json")

if __name__ == "__main__":
    main()
