

import matplotlib.pyplot as plt
import pandas as pd



# Load the Excel file
file_path = './datasetChange.xlsx'
data = pd.read_excel(file_path)

# Skip the first row (header) and use the first column as the index
data.index = data.iloc[:, 0]
data = data.iloc[:, 1:]

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

# Plot the data with a scientific journal style
plt.figure(figsize=(10, 6))

# Plot for 'Unet' and 'Proposed' methods with clearer lines and markers
plt.plot(data.columns, data.loc['U-Net'], marker='o', linestyle='-', color='blue', label='Unet[13]')
plt.plot(data.columns, data.loc['DAUnet'], marker='h', linestyle='-.', color='pink', label='DAUnet[21]')
plt.plot(data.columns, data.loc['TransUNet'], marker='p', linestyle='--', color='orange', label='TransUNet[15]')
plt.plot(data.columns, data.loc['zhao'], marker='<', linestyle='--', color='brown', label='Zhao[19]')
plt.plot(data.columns, data.loc['AttUnet'], marker='x', linestyle='-.', color='black', label='AttUnet[28]')
plt.plot(data.columns, data.loc['DenseNet'], marker='>', linestyle='--', color='gray', label='DenseNet[20]')
plt.plot(data.columns, data.loc['ResUNet++'], marker='^', linestyle='-.', color='red', label='ResUNet++[27]')
plt.plot(data.columns, data.loc['Proposed'], marker='s', linestyle='-', color='green', label='Proposed')




# Add labels, title, and legend with increased font size and clearer style
plt.xlabel('Dataset Percentage (%)', fontsize=16, fontweight='bold')
plt.ylabel('Dice Coefficient', fontsize=16, fontweight='bold')
# plt.title('Dice Coefficient of Different Methods vs Dataset Percentage', fontsize=16, fontweight='bold')
plt.legend(frameon=True, fontsize=16, edgecolor='black')

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim((0.72,0.8))
# Display the plot
plt.tight_layout()
plt.show()
