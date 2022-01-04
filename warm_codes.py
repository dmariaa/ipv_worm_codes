import os.path
import time

import cv2.cv2 as cv2
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
import flirimageextractor


class WarmCodesReader:
    positions = [
        (121, 142, 57, 114),  # 0
        (178, 142, 57, 57),  # 1
        (178, 199, 57, 57),  # 2
        (178, 256, 57, 57),  # 3
        (235, 142, 57, 57),  # 4
        (235, 199, 57, 57),  # 5
        (235, 256, 57, 57),  # 6
        (292, 142, 57, 57),  # 7
        (292, 199, 57, 57),  # 8
        (292, 256, 57, 57)
    ]

    ir_positions = [
        (54, 180, 70, 140),  # 0
        (124, 180, 70, 70),  # 1
        (124, 250, 70, 70),  # 2
        (124, 320, 70, 70),  # 3
        (194, 180, 70, 70),  # 4
        (194, 250, 70, 70),  # 5
        (194, 320, 70, 70),  # 6
        (264, 180, 70, 70),  # 7
        (264, 250, 70, 70),  # 8
        (264, 320, 70, 70),  # 9
    ]

    def __init__(self, data_file, intensity_factor=0.7, minimum_energy=1.):
        self.intensity_factor = intensity_factor
        self.minimum_energy = minimum_energy
        self.data_file = data_file
        self.extractor = flirimageextractor.FlirImageExtractor(palettes=[cm.jet, cm.bwr, cm.gist_ncar])
        self.data = None

        # init
        self.get_data()

    def get_data(self):
        self.data = pd.read_csv(self.data_file, sep=';', names=['file', 'digits'], skip_blank_lines=True,
                                keep_default_na=False)

    def read_flir(self, index):
        flir_file = self.data.iloc[index]['file']
        self.extractor.process_image(os.path.join('img', f"{flir_file}.jpg"))
        color = self.extractor.extract_embedded_image()
        thermal = self.extractor.extract_thermal_image()
        return color, thermal

    def normalize(self, data, range=1):
        normalized = (data - data.min()) / np.ptp(data)
        return normalized * range

    def thermal_mask(self, thermal):
        threshold = thermal.min() + ((thermal.max() - thermal.min()) * self.intensity_factor)
        outp = (thermal <= threshold)
        masked = thermal.copy()
        masked[outp] = 0.
        masked[~outp] = 1.
        return masked

    def get_cell_energy(self, thermal, block):
        mask = self.thermal_mask(thermal)
        thermal_masked = thermal * mask

        (x, y, w, h) = block
        key_block = thermal_masked[y:y + h + 1, x:x + w + 1]
        return np.sum(key_block) / (w * h)

    def get_energy_count(self, thermal):
        energies = np.zeros(10, dtype=float)

        for i, (x, y, w, h) in enumerate(self.ir_positions):
            energy = self.get_cell_energy(thermal, (x, y, w, h))
            energies[i] = energy if energy > self.minimum_energy else 0.

        return energies

    def get_pressed_buttons(self, index):
        color, thermal = self.read_flir(index)
        energy = self.get_energy_count(thermal)
        indexes = np.argwhere(energy > 0.).ravel()
        energies = energy[indexes]
        buttons_indexes = np.argsort(energies)
        buttons = indexes[buttons_indexes]
        return buttons, color, thermal

    def draw_reference_lines(self, image, use_ir_positions=True):
        if use_ir_positions:
            for (x, y, w, h) in self.ir_positions:
                image = cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=1)
        else:
            for (x, y, w, h) in self.positions:
                image = cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=1)

        return image

    def plot_sample(self, color, thermal):
        color = self.draw_reference_lines(color, use_ir_positions=False)
        thermal_lines = thermal.copy()
        thermal_lines = self.draw_reference_lines(thermal_lines)

        plt.figure(figsize=(30, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(color, cmap='gray')
        plt.subplot(1, 3, 2)
        plt.imshow(thermal_lines, cmap='jet')
        plt.subplot(1, 3, 3)
        plt.imshow(self.thermal_mask(thermal), cmap='gray')


reader = WarmCodesReader('datos.csv', intensity_factor=0.75, minimum_energy=1.)
# energy = reader.get_energy_count(0)
# print(energy)

# index = 18
# buttons, color, thermal = reader.get_pressed_buttons(index)
# reader.plot_sample(color, thermal)
# plt.suptitle(f"Index: {index}: File: {reader.data.iloc[index]['file']} Label: {reader.data.iloc[index]['digits']} "
#              f"Prediction: {buttons}")
# plt.show()

for index, data in reader.data.loc[reader.data['digits'] == ''].iterrows():
    buttons, color, thermal = reader.get_pressed_buttons(index)
    reader.plot_sample(color, thermal)
    plt.suptitle(f"Index: {index}: File: {reader.data.iloc[index]['file']} Label: {reader.data.iloc[index]['digits']} "
                 f"Prediction: {buttons}")
    plt.show()
