import os
import numpy as np
import matplotlib.pyplot as plt
import math

def create_confusion_matrix_plot(confusion_matrices, titles=None, class_labels=None, 
                                save_path=None, figsize_per_plot=(4, 4), 
                                title_fontsize=10, label_fontsize=8, tick_fontsize=6, 
                                text_fontsize=8, cmap='Blues', max_cols=3,
                                show_numbers=True, number_format="{:.1f}"):
    """
    Create a grid of confusion matrix plots with customizable parameters.
    
    Parameters:
    -----------
    confusion_matrices : list of 2D numpy arrays
        List of confusion matrices to plot
    titles : list of str, optional
        Titles for each confusion matrix. If None, uses "Matrix 1", "Matrix 2", etc.
    class_labels : list of str, optional
        Class labels for axes. If None, uses indices
    save_path : str, optional
        Path to save the figure. If None, doesn't save
    figsize_per_plot : tuple, optional
        Size of each subplot (width, height)
    title_fontsize : int, optional
        Font size for subplot titles
    label_fontsize : int, optional
        Font size for axis labels
    tick_fontsize : int, optional
        Font size for tick labels
    text_fontsize : int, optional
        Font size for numbers in cells
    cmap : str, optional
        Colormap for the confusion matrices
    max_cols : int, optional
        Maximum number of columns in the grid
    show_numbers : bool, optional
        Whether to show numbers in confusion matrix cells
    number_format : str, optional
        Format string for numbers in cells
    """
    
    num_matrices = len(confusion_matrices)
    if num_matrices == 0:
        print("No confusion matrices provided!")
        return
    
    # Generate default titles if none provided
    if titles is None:
        titles = [f"Matrix {i+1}" for i in range(num_matrices)]
    
    # Calculate grid dimensions
    cols = min(max_cols, num_matrices)
    rows = math.ceil(num_matrices / cols)
    
    # Create figure
    fig_width = figsize_per_plot[0] * cols
    fig_height = figsize_per_plot[1] * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    
    # Handle different axis configurations
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])
    
    # Plot each confusion matrix
    for idx, cm in enumerate(confusion_matrices):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        
        # Plot confusion matrix
        im = ax.imshow(cm, interpolation='nearest', cmap=getattr(plt.cm, cmap))
        
        # Set title and labels
        ax.set_title(titles[idx], fontsize=title_fontsize)
        ax.set_xlabel("Predicted", fontsize=label_fontsize)
        ax.set_ylabel("True", fontsize=label_fontsize)
        
        # Set tick labels
        if class_labels is not None:
            ax.set_xticks(range(len(class_labels)))
            ax.set_yticks(range(len(class_labels)))
            ax.set_xticklabels(class_labels, fontsize=tick_fontsize)
            ax.set_yticklabels(class_labels, fontsize=tick_fontsize)
        else:
            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        
        # Add numbers to cells if requested
        if show_numbers:
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    color = "white" if cm[i, j] > thresh else "black"
                    ax.text(j, i, number_format.format(cm[i, j]), 
                           ha="center", va="center", color=color, fontsize=text_fontsize)
    
    # Hide any unused subplots
    for idx in range(num_matrices, rows * cols):
        r = idx // cols
        c = idx % cols
        fig.delaxes(axes[r, c])
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig("plots/augmented_emotion_embedding_constrained.png", dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to: {save_path}")
    
    plt.show()
    return fig, axes

# Example usage functions
def example_load_from_folder():
    """Example: Load confusion matrices from .npy files in a folder (original style)"""
    folder = "noun_verb_raweeg_model_evaluation_results"
    files = [f for f in os.listdir(folder) if f.endswith("_confusion_matrix.npy")]
    files.sort()
    
    confusion_matrices = []
    titles = []
    
    for fname in files:
        cm = np.load(os.path.join(folder, fname))
        confusion_matrices.append(cm)
        titles.append(fname.replace("_confusion_matrix.npy", ""))
    
    create_confusion_matrix_plot(
        confusion_matrices=confusion_matrices,
        titles=titles,
        class_labels=["Noun", "Verb"],  # Customize as needed
        save_path="plots/testing_raw_eeg_model_cm_results.png",
        figsize_per_plot=(4, 4),
        title_fontsize=8,
        label_fontsize=6,
        tick_fontsize=4,
        text_fontsize=6
    )

def example_custom_matrices():
    """Example: Create custom confusion matrices with editable numbers"""
    
    # Define custom confusion matrices with your own numbers
    cm1 = np.array([[85, 12, 3],
                    [8, 91, 1],
                    [5, 2, 93]])
    
    cm2 = np.array([[78, 15, 7],
                    [12, 88, 0],
                    [10, 5, 85]])
    
    cm3 = np.array([[92, 6, 2],
                    [4, 94, 2],
                    [3, 1, 96]])
    
    # Binary classification example
    cm4 = np.array([[156, 24],
                    [18, 162]])
    
    confusion_matrices = [cm1, cm2, cm3, cm4]
    titles = ["Model A (3-class)", "Model B (3-class)", "Model C (3-class)", "Binary Model"]
    
    # Plot with custom settings
    create_confusion_matrix_plot(
        confusion_matrices=confusion_matrices,
        titles=titles,
        class_labels=["Negative", "Neutral", "Positive"],  # Will only apply to first 3 matrices
        save_path="custom_confusion_matrices.png",
        figsize_per_plot=(3.5, 3.5),
        title_fontsize=10,
        label_fontsize=8,
        tick_fontsize=6,
        text_fontsize=7,
        cmap='Blues',
        max_cols=2,  # 2 columns instead of 3
        number_format="{:.0f}"  # No decimals for count data
    )

def example_percentage_matrices():
    """Example: Create confusion matrices showing percentages"""
    
    # Confusion matrices as percentages
    cm1 = np.array([[0.85, 0.12, 0.03],
                    [0.08, 0.91, 0.01],
                    [0.05, 0.02, 0.93]])
    
    cm2 = np.array([[0.78, 0.22],
                    [0.15, 0.85]])
    
    confusion_matrices = [cm1, cm2]
    titles = ["3-class Accuracy", "Binary Accuracy"]
    
    create_confusion_matrix_plot(
        confusion_matrices=confusion_matrices,
        titles=titles,
        class_labels=None,  # Use default numeric labels
        save_path="percentage_confusion_matrices.png",
        cmap='Greens',
        number_format="{:.2f}",  # Show 2 decimal places for percentages
        text_fontsize=8
    )

def create_single_editable_matrix():
    """Create a single confusion matrix with easily editable numbers"""
    
    # EDIT THESE NUMBERS TO CUSTOMIZE YOUR CONFUSION MATRIX
    confusion_data = [
        [697, 33, 21],    # True class 0: 120 correct, 8 confused with class 1, 2 with class 2
        [29, 717, 22],   # True class 1: 5 confused with class 0, 115 correct, 10 with class 2  
        [26, 28, 942]     # True class 2: 3 confused with class 0, 7 with class 1, 140 correct
    ]
    
    cm = np.array(confusion_data)
    
    create_confusion_matrix_plot(
        confusion_matrices=[cm],
        titles=["Emotion Confusion Matrix"],
        class_labels=["Negative", "Neutral", "Positive"],  # EDIT THESE CLASS NAMES
        save_path="my_custom_confusion_matrix.png",  # EDIT SAVE PATH
        figsize_per_plot=(5, 5),
        cmap='Blues',  # OPTIONS: 'Blues', 'Greens', 'Reds', 'Purples', 'Oranges'
        number_format="{:.0f}"  # No decimals for count data
    )

if __name__ == "__main__":
    # Uncomment the example you want to run:
    
    # 1. Original style - load from folder
    # example_load_from_folder()
    
    # 2. Custom matrices with your own numbers
    create_single_editable_matrix()
    
    # 3. Percentage/probability matrices
    # example_percentage_matrices()
    
    # 4. Single editable matrix
    # create_single_editable_matrix()