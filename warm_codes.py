import os.path

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
import flirimageextractor


def read_flir(file):
    ex = flirimageextractor.FlirImageExtractor(palettes=[cm.jet, cm.bwr, cm.gist_ncar])
    ex.process_image(os.path.join('img', f"{file}.jpg"))
    color = ex.extract_embedded_image()
    thermal = ex.extract_thermal_image()
    return color, thermal


def thermal_mask(thermal):
    out = np.where(thermal < 9)
    thermal[out] = 0
    return thermal


def plot_sample(color, thermal, file, digits):
    color, thermal = read_flir(file)
    plt.subplot(1, 2, 1)
    plt.imshow(color, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(thermal, cmap='jet')
    plt.suptitle(f"File: {file} Label: {digits}")
    plt.show()


def plot_fft(sample):
    pass


data = pd.read_csv('datos.csv', sep=';', names=['file', 'digits'],
                   skip_blank_lines=True, keep_default_na=False)

file = data.iloc[9]['file']
digits = data.iloc[9]['digits']
rgb, thermal = read_flir(file)

plot_sample(rgb, thermal, file, digits)

mask = thermal_mask(thermal)
plt.imshow(mask)
plt.show()