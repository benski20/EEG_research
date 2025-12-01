
import os
import glob
import h5py
import pandas as pd
from tensorflow.keras.models import load_model

# Directory to start the recursive search (update to your database root path)
root_dir = '.'  # Replace with '/path/to/your/database' or leave as '.' for current directory

# List of model filenames or patterns to exclude (e.g., exact filenames or partial matches)
excluded_models = ['exclude_this_model.h5', 'another_to_skip.h5']  # Add more as needed

# Find all .h5 files recursively, excluding specified models
model_files = []
for dirpath, _, filenames in os.walk(root_dir):
    for file in filenames:
        if file.endswith('.h5') and not any(excl in file for excl in excluded_models):
            model_files.append(os.path.join(dirpath, file))

# List to hold extracted data for each model
data = []

for file_path in model_files:
    model_name = os.path.basename(file_path)
    try:
        # Check if the file has a valid model configuration
        with h5py.File(file_path, 'r') as f:
            if 'model_config' not in f.attrs:
                print(f"Skipping {model_name}: No 'model_config' attribute found (may be weights-only file).")
                continue
        
        # Load the full model (architecture + weights)
        model = load_model(file_path, compile=False)  # compile=False to avoid loading optimizer state
        
        # Extract key information
        input_shape = model.input_shape
        num_layers = len(model.layers)
        num_params = model.count_params()  # Total trainable parameters
        # Architecture summary: list of layer types and output shapes
        layers_summary = '; '.join([
            f"{layer.name} ({layer.__class__.__name__}): Output Shape {layer.output_shape}"
            for layer in model.layers
        ])
        
        # Append to data list
        data.append({
            'Model Name': model_name,
            'Input Shape': str(input_shape),
            'Number of Layers': num_layers,
            'Number of Parameters': num_params,
            'Architecture Summary': layers_summary
        })
        
    except Exception as e:
        print(f"Error processing {model_name}: {str(e)}")
        continue

# Create a pandas DataFrame for the table
if data:
    df = pd.DataFrame(data)
    
    # Generate LaTeX table for scientific paper
    latex_table = df.to_latex(
        index=False,
        escape=False,  # Prevent escaping special characters
        longtable=True,  # Use longtable for multi-page tables
        column_format='lcccc',  # Left-align Model Name, center others
        header=['Model Name', 'Input Shape', 'Number of Layers', 'Number of Parameters', 'Architecture Summary']
    )
    
    # Create a complete LaTeX document
    latex_document = r"""
\documentclass{article}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{geometry}
\geometry{a4paper, margin=1in}
\usepackage{amsmath}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}

\begin{document}

\section*{Model Architectures}
The following table summarizes the architectures of the neural network models used in this study.

""" + latex_table + r"""

\end{document}
"""
    
    # Save LaTeX document to a file
    with open('model_architectures.tex', 'w') as f:
        f.write(latex_document)
    
    print("LaTeX table generated and saved as 'model_architectures.tex'.")
    print(f"Found and processed {len(data)} valid models.")
    
    # Optional: Export to CSV for further editing
    df.to_csv('model_architectures.csv', index=False)
    print("Data also exported to 'model_architectures.csv' for reference.")
else:
    print("No valid .h5 models found or all excluded.")
