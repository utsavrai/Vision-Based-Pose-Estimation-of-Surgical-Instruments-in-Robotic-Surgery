import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def extract_tb_data(logdir):
    scalar_data = {}

    # Walk through all subdirectories and files in the log directory
    for root, _, files in os.walk(logdir):
        for file in files:
            if file.startswith("events.out"):
                event_file = os.path.join(root, file)
                
                # Extract the run name from the directory structure
                run_name = os.path.relpath(root, logdir)
                
                # Initialize a dictionary for the current run
                if run_name not in scalar_data:
                    scalar_data[run_name] = {}
                
                # Use event_accumulator to read the event file
                ea = event_accumulator.EventAccumulator(event_file)
                ea.Reload()
                
                for tag in ea.Tags()["scalars"]:
                    # Only process tags containing 'loss'
                    if 'loss' in tag:
                        if tag not in scalar_data[run_name]:
                            scalar_data[run_name][tag] = {'step': [], 'value': []}
                        
                        events = ea.Scalars(tag)
                        for event in events:
                            scalar_data[run_name][tag]['step'].append(event.step)
                            scalar_data[run_name][tag]['value'].append(event.value)
    
    # Sort and remove duplicates
    for run_name in scalar_data:
        for tag in scalar_data[run_name]:
            steps_values = list(zip(scalar_data[run_name][tag]['step'], scalar_data[run_name][tag]['value']))
            steps_values = sorted(set(steps_values))  # Remove duplicates and sort
            scalar_data[run_name][tag]['step'], scalar_data[run_name][tag]['value'] = zip(*steps_values)
    
    return scalar_data


def plot_scalar_data(scalar_data, line_width=2):
    tags_runs = {}

    # Group data by tags
    for run, run_data in scalar_data.items():
        for tag, data in run_data.items():
            if tag not in tags_runs:
                tags_runs[tag] = []
            tags_runs[tag].append((run, data))
    
    num_tags = len(tags_runs)
    cols = 2  # Number of columns in the subplot grid
    rows = (num_tags + cols - 1) // cols  # Calculate the number of rows needed
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    font_size = 14  # Define a font size variable for easy adjustments
    
    for ax, (tag, runs_data) in zip(axes, tags_runs.items()):
        for run, data in runs_data:
            ax.plot(data['step'], data['value'], label=run, linewidth=line_width)
        ax.set_xlabel('Step', fontsize=font_size)
        ax.set_ylabel('Value', fontsize=font_size)
        ax.set_yscale('log')  # Set the y-axis to logarithmic scale
        ax.legend(fontsize=font_size)
        ax.set_title(f'{tag}', fontsize=font_size)
    
    # Remove any unused subplots
    for ax in axes[num_tags:]:
        fig.delaxes(ax)
    
    plt.tight_layout()  # Automatically adjusts the padding
    # plt.subplots_adjust(hspace=0.4, wspace=0.4)  # Increase vertical/horizontal space
    plt.subplots_adjust(hspace=0.3)  # Add vertical space between subplots
    plt.savefig("fig.png", dpi=300)
    plt.show()

# Path to the log directory
logdir = 'output/gdrn/lnd/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lnd/tb'

# Extract data
scalar_data = extract_tb_data(logdir)

# Plot data with increased line thickness
plot_scalar_data(scalar_data, line_width=2)


# import os
# import time
# from tensorboard.backend.event_processing import event_accumulator

# def extract_all_tags_with_retries(logdir, max_retries=3, delay=1):
#     all_tags = set()  # Use a set to store unique tags
    
#     # Walk through all subdirectories and files in the log directory
#     for root, _, files in os.walk(logdir):
#         for file in files:
#             if file.startswith("events.out"):
#                 event_file = os.path.join(root, file)
                
#                 # Use event_accumulator to read the event file
#                 ea = event_accumulator.EventAccumulator(event_file)
                
#                 # Retry loading if necessary
#                 retries = 0
#                 while retries < max_retries:
#                     try:
#                         ea.Reload()
#                         break  # Exit the retry loop if successful
#                     except Exception as e:
#                         print(f"Retrying... {retries + 1}/{max_retries}. Error: {e}")
#                         retries += 1
#                         time.sleep(delay)
                
#                 # Add all scalar tags to the set
#                 all_tags.update(ea.Tags()["scalars"])
    
#     return all_tags

# # Path to the log directory
# logdir = 'output/gdrn/lnd/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lnd/tb'

# # Extract and print all unique tags with retries
# all_tags = extract_all_tags_with_retries(logdir)

# print("Available Tags in the Log (after retries):")
# for tag in sorted(all_tags):
#     print(tag)
