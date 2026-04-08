#this file is for converting the output signal file type from .npy to .txt#
#this is crucial in order to check the EVM of the interpolated signal#
import numpy as np
import os

base_path = '/content/drive/MyDrive/DEEP LEARNING PROJECT/'

# 1. Load the binary file (Now using the v2 name)
data = np.load(base_path + 'upsample_signal_v2.npy')

# 2. Save as a text file
# To reach ~17MB, we increase precision to 14 decimal places ('%.14f')
# We use a tab delimiter to match the 'zizo' format
np.savetxt(base_path + 'upsample_signal_v2.txt', data, fmt='%.14f', delimiter='\t')

# 3. Verification
file_size_mb = os.path.getsize(base_path + 'upsample_signal_v2.txt') / (1024 * 1024)
print(f"Conversion complete!")
print(f"Total Samples: {data.shape[0]}")
print(f"New File Size: {file_size_mb:.2f} MB")
