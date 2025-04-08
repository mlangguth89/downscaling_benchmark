import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from argparse import ArgumentParser

# Argument parser for CLI
parser = ArgumentParser(description='Plotting script.')
parser.add_argument('--output-dir', type=str, required=True, help="Folder to store the plots")
parser.add_argument('--model-dir', type=str, required=True, help="Folder containing model logs")
parser.add_argument('--model-types', type=str, required=True, help="Comma-separated list of processed model types")
parser.add_argument('--datasets', type=str, required=True, help="Comma-separated list of processed dataset names")
args = parser.parse_args()

output_dir = args.output_dir
model_dir = args.model_dir
model_types = [model_type.strip() for model_type in args.model_types.split(",")]
datasets = [dataset.strip() for dataset in args.datasets.split(",")]


def extract_metrics_from_log(file_path):
    """Extracts RMSE and Gradient Amplitude from a .log file."""
    rmse = None
    grad_amplitude = None

    with open(file_path, 'r') as file:
        for line in file:
            if "Globally averaged rmse" in line:
                print(line)
                parts = line.split('INFO:')[1].split(": ")
                rmse_value = float(parts[1].split(" ")[0])  
                rmse_value_std = float(parts[2].split(" ")[0]) 
                rmse = (rmse_value, rmse_value_std)
    
            elif "Globally averaged grad_amplitude" in line:
                print(line)
                parts = line.split('INFO:')[1].split(": ")
                grad_amplitude_value = float(parts[1].split(" ")[0])  
                grad_amplitude_value_std = float(parts[2].split(" ")[0])  
                grad_amplitude = (grad_amplitude_value, grad_amplitude_value_std)

    return rmse, grad_amplitude


def create_dataframe_from_logs(model_dir):
    """Scans all subdirectories in model_dir and extracts values from relevant .log files."""
    data = {'model': [], 'model_number': [], 'rmse_global': [], 'grad_amplitude_global': []}

    for model_name in os.listdir(model_dir):
        model_path = os.path.join(model_dir, model_name)

        if os.path.isdir(model_path):  # Check if it is a directory
            for file_name in os.listdir(model_path):
                if file_name.startswith(f"postprocessing_") and file_name.endswith(".log"):
                    file_path = os.path.join(model_path, file_name)

                    rmse, grad_amplitude = extract_metrics_from_log(file_path)
                    if rmse is not None and grad_amplitude is not None:
                        model_index_match = re.search(r'_(\d+)\.log$', file_name)
                        model_index = model_index_match.group(1) if model_index_match else "unknown"

                        data['model'].append(model_name)
                        data['model_number'].append(model_index)
                        data['rmse_global'].append(rmse)
                        data['grad_amplitude_global'].append(grad_amplitude)

    return pd.DataFrame(data)


# Create DataFrame
df = create_dataframe_from_logs(model_dir)

print(str(df))

# Assign model categories
df['model_type'] = df['model'].apply(
    lambda x: next((model_type for model_type in model_types if model_type in x), 'unknown')
)

df['dataset'] = df['model'].apply(
    lambda x: next((dataset for dataset in datasets if dataset in x), 'unknown')
)

# Filter out entries with 'unknown' in 'model_type' or 'dataset'
df = df[(df['model_type'] != 'unknown') & (df['dataset'] != 'unknown')]

# Remove unnecessary columns
df = df.drop(columns=['model'])
df_lean = df.drop(columns=['model_number'])

#print(str(df_lean))

# Function to calculate the mean of tuples
def tuple_mean(tuples):
    return tuple(sum(x) / len(x) for x in zip(*tuples))

# Identify columns containing tuples
tuple_columns = df_lean.drop(columns=['model_type', 'dataset']).columns

# Define aggregation method for each tuple column
agg_dict = {col: tuple_mean for col in tuple_columns}

# Calculate average values
avg_df = df_lean.groupby(['model_type', 'dataset']).agg(agg_dict).reset_index()


# Predefined lists of colors and markers
available_colors = plt.cm.tab10.colors  # You can also use a custom list like ['blue', 'red', 'green', ...]
available_markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p', '*', 'x']

# Assign colors dynamically to model types
colors = {model: available_colors[i % len(available_colors)] for i, model in enumerate(model_types)}

# Assign markers dynamically to datasets
markers = {dataset: available_markers[i % len(available_markers)] for i, dataset in enumerate(datasets)}

# Define output PDF file
pdf_path = os.path.join(output_dir, "compare_plots.pdf")

with PdfPages(pdf_path) as pdf:
    for dataset_name in datasets:
        plt.figure(figsize=(10, 6))

        # Scatter plot for each model_type & dataset
        for model_type in df['model_type'].unique():
            for dataset in df['dataset'].unique():
                if dataset == dataset_name:
                    temp_df = df[(df['model_type'] == model_type) & (df['dataset'] == dataset)]
                    
                    # Scatter plot
                    plt.scatter(
                        temp_df['rmse_global'].apply(lambda x: x[0]), 
                        temp_df['grad_amplitude_global'].apply(lambda x: x[0]), 
                        s=20, 
                        label=f'{model_type}', 
                        color=colors.get(model_type, 'black'),
                        marker=markers.get(dataset, 'o')
                    )
                    
                    # Annotate points
                    for i, row in temp_df.iterrows():
                        plt.annotate(
                            row['model_number'],  
                            (row['rmse_global'][0], row['grad_amplitude_global'][0]),  
                            textcoords="offset points", 
                            xytext=(0, 5), 
                            ha='center',
                            fontsize=6
                        )
        
        # Plot average values
        for index, row in avg_df.iterrows():
            if row['dataset'] == dataset_name:
                plt.scatter(row['rmse_global'][0], row['grad_amplitude_global'][0], 
                            s=100,  
                            color=colors.get(row['model_type'], 'black'), 
                            marker=markers.get(row['dataset'], 'o'))
        
        # Axis labels and title
        plt.xlabel('RMSE')
        plt.ylabel('Gradient Amplitude')
        plt.title(f'RMSE vs. Gradient Amplitude ({dataset_name})')
        plt.legend(fontsize='small')
        plt.grid(True)
        
        # Save current figure to PDF
        pdf.savefig()
        plt.close()
    
    print(f"All plots saved in {pdf_path}")
