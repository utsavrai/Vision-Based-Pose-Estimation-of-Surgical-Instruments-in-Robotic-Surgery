import matplotlib.pyplot as plt
import numpy as np

# Data for all methods
methods = ['GDR-Net', 'PVNet', 'FoundationPose', 'SAM-6D Mask-RCNN', 'SAM-6D SAM', 'OVE6D', 'Megapose']
datasets = ['Non Occluded', 'Occluded']

data = {
    'FoundationPose': {
        'Non Occluded': {
            'ADD': [0.00, 14.39, 29.76, 35.37, 35.85, 35.85, 36.10, 36.10, 89.02, 92.93, 93.17],
            '2D Projection': [0.00, 27.80, 35.61, 36.10, 36.10, 36.10, 38.78, 42.93, 58.54, 74.39, 86.10],
        },
        'Occluded': {
            'ADD': [0.00, 2.39, 25.78, 30.07, 30.79, 31.26, 31.26, 31.26, 86.63, 89.98, 90.93],
            '2D Projection': [0.00, 3.58, 27.92, 31.50, 31.74, 31.98, 41.77, 73.99, 91.17, 91.89, 92.60],
        }
    },
    'SAM-6D Mask-RCNN': {
        'Non Occluded': {
            'ADD': [0.00, 14.39, 28.78, 31.46, 33.17, 35.61, 37.80, 39.76, 89.27, 91.71, 93.41],
            '2D Projection': [0.00, 22.20, 32.20, 32.93, 33.90, 34.88, 37.80, 42.20, 56.34, 72.93, 85.61],
        },
        'Occluded': {
            'ADD': [0.00, 6.92, 22.91, 29.83, 32.46, 34.13, 35.32, 36.75, 84.73, 91.65, 96.18],
            '2D Projection': [0.00, 29.83, 34.84, 35.32, 35.80, 36.75, 45.11, 71.84, 91.17, 94.51, 97.37],
        }
    },
    'SAM-6D SAM': {
        'Non Occluded': {
            'ADD': [0.00, 0.00, 0.24, 0.24, 0.24, 1.71, 6.59, 12.44, 16.10, 19.02, 22.68],
            '2D Projection': [0.00, 0.00, 0.24, 0.49, 0.49, 0.49, 2.93, 4.63, 8.78, 11.95, 14.15],
        },
        'Occluded': {
            'ADD': [0.00, 0.00, 0.00, 0.00, 0.00, 0.24, 0.24, 0.48, 0.95, 1.19, 1.43],
            '2D Projection': [0.00, 0.00, 0.24, 0.48, 0.48, 0.48, 0.48, 0.72, 0.95, 1.67, 1.91],
        }
    },
    'OVE6D': {
        'Non Occluded': {
            'ADD': [0.00, 0.00, 0.00, 0.00, 3.17, 10.24, 19.76, 29.27, 71.46, 89.51, 93.41],
            '2D Projection': [0.00, 0.00, 0.00, 0.00, 1.71, 8.78, 15.12, 27.07, 44.88, 62.93, 75.85],
        },
        'Occluded': {
            'ADD': [0.00, 0.00, 0.00, 0.00, 0.00, 1.45, 17.59, 42.17, 66.99, 90.60, 94.22],
            '2D Projection': [0.00, 0.00, 0.00, 0.24, 6.51, 22.89, 35.18, 69.64, 90.60, 97.83, 99.04],
        }
    },
    'Megapose': {
        'Non Occluded': {
            'ADD': [0.00, 0.00, 0.00, 0.24, 0.48, 0.72, 1.69, 2.41, 5.06, 8.67, 12.29],
            '2D Projection': [0.00, 0.24, 12.53, 26.51, 38.07, 48.67, 61.93, 74.70, 78.07, 83.13, 90.12],
        },
        'Occluded': {
            'ADD': [0.00, 0.00, 0.00, 0.00, 0.00, 0.24, 1.46, 1.71, 3.17, 7.80, 11.46],
            '2D Projection': [0.00, 0.24, 0.98, 3.90, 5.37, 5.85, 9.76, 14.88, 25.61, 37.56, 45.12],
        }
    },
    'GDR-Net': {
        'Non Occluded': {
            'ADD': [0.00, 87.89, 98.79, 99.03, 99.52, 99.52, 99.76, 99.76, 99.76, 99.76, 99.76],
            '2D Projection': [0.00, 99.76, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00],
        },
        'Occluded': {
            'ADD': [0.00, 84.48, 96.06, 97.78, 98.28, 98.28, 98.28, 98.28, 99.01, 99.01, 99.01],
            '2D Projection': [0.00, 96.06, 96.80, 96.80, 97.04, 97.54, 97.78, 98.28, 98.28, 98.28, 98.28],
        }
    },
    'PVNet': {
        'Non Occluded': {
            'ADD': [0.00, 65.61, 90.49, 96.83, 98.05, 98.54, 98.78, 98.78, 98.78, 98.78, 98.78],
            '2D Projection': [0.00, 98.54, 98.78, 99.02, 99.27, 99.51, 99.51, 99.76, 100.00, 100.00, 100.00],
        },
        'Occluded': {
            'ADD': [0.00, 55.61, 87.35, 93.79, 95.70, 96.42, 97.14, 97.14, 97.61, 97.61, 97.85],
            '2D Projection': [0.00, 98.09, 98.81, 99.05, 99.05, 99.05, 99.05, 99.05, 99.05, 99.05, 99.05],
        }
    },
}

# Set up thresholds
thresholds_add = np.arange(0.0, 0.55, 0.05)  # ADD thresholds from 0% to 50%
thresholds_2d = np.arange(0, 55, 5)  # 2D Projection thresholds from 0 to 50 pixels

# Create 4 subplots: ADD for Non-Occluded, ADD for Occluded, 2D Projection for Non-Occluded, and 2D Projection for Occluded

# # ADD Non Occluded
# plt.figure(figsize=(10, 6))
# for method in methods:
#     plt.plot(thresholds_add, data[method]['Non Occluded']['ADD'], label=method)
# plt.title('ADD - Non Occluded')
# plt.xlabel('Threshold (%)')
# plt.ylabel('Accuracy (%)')
# plt.grid(True)
# plt.ylim([0, 100])
# plt.legend()
# plt.show()

# # ADD Occluded
# plt.figure(figsize=(10, 6))
# for method in methods:
#     plt.plot(thresholds_add, data[method]['Occluded']['ADD'], label=method)
# plt.title('ADD - Occluded')
# plt.xlabel('Threshold (%)')
# plt.ylabel('Accuracy (%)')
# plt.grid(True)
# plt.ylim([0, 100])
# plt.legend()
# plt.show()

# # 2D Projection Non Occluded
# plt.figure(figsize=(10, 6))
# for method in methods:
#     plt.plot(thresholds_2d, data[method]['Non Occluded']['2D Projection'], label=method)
# plt.title('2D Projection - Non Occluded')
# plt.xlabel('Threshold (pixels)')
# plt.ylabel('Accuracy (%)')
# plt.grid(True)
# plt.ylim([0, 100])
# plt.legend()
# plt.show()

# # 2D Projection Occluded
# plt.figure(figsize=(10, 6))
# for method in methods:
#     plt.plot(thresholds_2d, data[method]['Occluded']['2D Projection'], label=method)
# plt.title('2D Projection - Occluded')
# plt.xlabel('Threshold (pixels)')
# plt.ylabel('Accuracy (%)')
# plt.grid(True)
# plt.ylim([0, 100])
# plt.legend()
# plt.show()


# Define unique line styles and markers for each method
line_styles = {
    'FoundationPose': {'linestyle': '--', 'marker': 'o', 'linewidth': 2, 'markersize': 8},
    'SAM-6D Mask-RCNN': {'linestyle': '--', 'marker': '^', 'linewidth': 2, 'markersize': 8},
    'SAM-6D SAM': {'linestyle': '--', 'marker': 'v', 'linewidth': 2, 'markersize': 8},
    'OVE6D': {'linestyle': '--', 'marker': 's', 'linewidth': 2, 'markersize': 8},
    'Megapose': {'linestyle': '--', 'marker': 'D', 'linewidth': 2, 'markersize': 8},
    'GDR-Net': {'linestyle': '--', 'marker': '>', 'linewidth': 2, 'markersize': 8},  # New marker
    'PVNet': {'linestyle': '--', 'marker': 'd', 'linewidth': 2, 'markersize': 8},  # New marker
}

# ADD Non Occluded
plt.figure(figsize=(10, 6))
for method in methods:
    plt.plot(thresholds_add, data[method]['Non Occluded']['ADD'], 
             label=method, 
             linestyle=line_styles[method]['linestyle'], 
             marker=line_styles[method]['marker'], 
             linewidth=line_styles[method]['linewidth'], 
             markersize=line_styles[method]['markersize'], 
             markevery=1)  # Mark every 1 unit (customize interval if needed)
    
# Set title, labels and tick font sizes
plt.title('ADD - Non Occluded', fontsize=18)  # Title font size
plt.xlabel('Threshold (% of Diameter)', fontsize=16)  # X-axis label font size
plt.ylabel('Accuracy (%)', fontsize=16)  # Y-axis label font size
plt.xticks(fontsize=12)  # X-axis tick font size
plt.yticks(fontsize=12)  # Y-axis tick font size
plt.legend()
plt.grid(True)
plt.ylim([0, 110])

# Save the figure
plt.savefig('add_non_occluded.png', dpi=300, bbox_inches='tight')  # Save as PNG, high quality (300 DPI)
plt.show()
# ADD Occluded
plt.figure(figsize=(10, 6))
for method in methods:
    plt.plot(thresholds_add, data[method]['Occluded']['ADD'], 
             label=method, 
             linestyle=line_styles[method]['linestyle'], 
             marker=line_styles[method]['marker'], 
             linewidth=line_styles[method]['linewidth'], 
             markersize=line_styles[method]['markersize'], 
             markevery=1)  # Mark every 1 unit (customize interval if needed)
plt.title('ADD - Occluded', fontsize=18)  # Title font size
plt.xlabel('Threshold (% of Diameter)', fontsize=16)  # X-axis label font size
plt.ylabel('Accuracy (%)', fontsize=16)  # Y-axis label font size
plt.xticks(fontsize=12)  # X-axis tick font size
plt.yticks(fontsize=12)  # Y-axis tick font size
plt.legend()
plt.grid(True)
plt.ylim([0, 110])
plt.savefig('add_occluded.png', dpi=300, bbox_inches='tight')  # Save as PNG, high quality (300 DPI)
plt.show()
# 2D Projection Non Occluded
plt.figure(figsize=(10, 6))
for method in methods:
    plt.plot(thresholds_2d, data[method]['Non Occluded']['2D Projection'], 
             label=method, 
             linestyle=line_styles[method]['linestyle'], 
             marker=line_styles[method]['marker'], 
             linewidth=line_styles[method]['linewidth'], 
             markersize=line_styles[method]['markersize'], 
             markevery=1)  # Mark every 1 unit (customize interval if needed)

plt.title('2D Projection - Non Occluded', fontsize=18)  # Title font size
plt.xlabel('Threshold (pixels)', fontsize=16)  # X-axis label font size
plt.ylabel('Accuracy (%)', fontsize=16)  # Y-axis label font size
plt.xticks(fontsize=12)  # X-axis tick font size
plt.yticks(fontsize=12)  # Y-axis tick font size
plt.legend()
plt.grid(True)
plt.ylim([0, 110])


# Save the figure
plt.savefig('2d_non_occluded.png', dpi=300, bbox_inches='tight')  # Save as PNG, high quality (300 DPI)
plt.show()
# 2D Projection Occluded
plt.figure(figsize=(10, 6))
for method in methods:
    plt.plot(thresholds_2d, data[method]['Occluded']['2D Projection'], 
             label=method, 
             linestyle=line_styles[method]['linestyle'], 
             marker=line_styles[method]['marker'], 
             linewidth=line_styles[method]['linewidth'], 
             markersize=line_styles[method]['markersize'], 
             markevery=1)  # Mark every 1 unit (customize interval if needed)
plt.title('2D Projection - Occluded', fontsize=18)  # Title font size
plt.xlabel('Threshold (pixels)', fontsize=16)  # X-axis label font size
plt.ylabel('Accuracy (%)', fontsize=16)  # Y-axis label font size
plt.xticks(fontsize=12)  # X-axis tick font size
plt.yticks(fontsize=12)  # Y-axis tick font size
plt.legend()
plt.grid(True)
plt.ylim([0, 110])
# Save the figure
plt.savefig('2d_occluded.png', dpi=300, bbox_inches='tight')  # Save as PNG, high quality (300 DPI)
plt.show()