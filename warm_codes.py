import csv
import os.path
import time

import cv2.cv2 as cv2
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
import flirimageextractor


class WarmCodesReader:
    """
    Reader for warm codes.
    Reads a flir file containing key typing on numeric keyboard and detects keys typed.
    """
    positions = np.array([
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
    ])

    ir_positions = np.array([
        (59, 185, 60, 130),  # 0
        (129, 185, 60, 60),  # 1
        (129, 255, 60, 60),  # 2
        (129, 325, 60, 60),  # 3
        (199, 185, 60, 60),  # 4
        (199, 255, 60, 60),  # 5
        (199, 325, 60, 60),  # 6
        (269, 185, 60, 60),  # 7
        (269, 255, 60, 60),  # 8
        (269, 325, 60, 60),  # 9
    ])

    def __init__(self, data_file, intensity_factor=0.7, minimum_energy=1., thresholding='own'):
        """
        Initializes the warm codes reader
        :param data_file: csv file describing data
        :param intensity_factor: factor of intensity capture for own thresholder
        :param minimum_energy: minimum energy to accept as key
        :param thresholding: 'own' to use own thresholder, 'otsu' to use otsu_multilevel
        """
        self.intensity_factor = intensity_factor
        self.minimum_energy = minimum_energy
        self.data_file = data_file
        self.extractor = flirimageextractor.FlirImageExtractor(palettes=[cm.jet, cm.bwr, cm.gist_ncar])
        self.data = None
        self.thresholder = self.thermal_mask if thresholding == 'own' else self.thermal_mask_otsu

        self.x_min = self.ir_positions[:, 0].min()
        self.y_min = self.ir_positions[:, 1].min()
        self.x_max = (self.ir_positions[:, 0] + self.ir_positions[:, 2]).max()
        self.y_max = (self.ir_positions[:, 1] + self.ir_positions[:, 3]).max()

        # init
        self.get_data()

    def get_data(self):
        """
        Reads data
        :return:
        """
        self.data = pd.read_csv(self.data_file, sep=';', names=['file', 'digits'], skip_blank_lines=True,
                                keep_default_na=False)

    def read_flir(self, index):
        """
        Reads flir file
        :param index: index in data of file to read
        :return:
        """
        flir_file = self.data.iloc[index]['file']
        self.extractor.process_image(os.path.join('img', f"{flir_file}.jpg"))
        color = self.extractor.extract_embedded_image()
        thermal = self.extractor.extract_thermal_image()
        return color, thermal

    def normalize(self, data):
        """
        Normalizes data in 0-1 range
        :param data:
        :param range:
        :return:
        """
        normalized = (data - np.min(data)) / np.ptp(data)
        return normalized

    def thermal_mask(self, thermal):
        """
        Return thermal image mask, own method
        :param thermal: thermal image
        :return: mask
        """
        masked = np.zeros(thermal.shape)
        t = thermal[self.y_min:self.y_max + 1, self.x_min:self.x_max + 1]
        threshold = t.min() + ((t.max() - t.min()) * self.intensity_factor)
        outp = (thermal > threshold)
        masked[outp] = 1.
        return masked

    def thermal_mask_otsu(self, thermal):
        """
        Returns thermal image mask, otsu multilevel method
        :param thermal: thermal image
        :return: mask
        """
        from skimage.filters import threshold_multiotsu, threshold_otsu
        threshold = threshold_multiotsu(thermal, classes=4)
        outp = (thermal > threshold[-1])
        masked = np.zeros(thermal.shape)
        masked[outp] = 1.

        masked = cv2.erode(masked, kernel=np.ones((3, 3)), iterations=2)
        return masked

    def get_cell_energy(self, thermal_masked, block):
        """
        Calculates energy for a cell in thermal image
        :param thermal: thermal image
        :param block: top, left, width, height block
        :return: energy (sum of values / area of block)
        """
        (x, y, w, h) = block
        key_block = thermal_masked[y:y + h + 1, x:x + w + 1]
        return np.sum(key_block) / (w * h)

    def get_energy_count(self, thermal):
        """
        Calculates energy for all cells corresponding to digit keys
        :param thermal: thermal image
        :return: array of energies by key, normalized in 0-1 range, all values below self.minimum_energy
        ar dropped to 0.
        """
        mask = self.thresholder(thermal)
        thermal_masked = thermal * mask

        energies = np.zeros(10, dtype=float)

        for i, (x, y, w, h) in enumerate(self.ir_positions):
            energy = self.get_cell_energy(thermal_masked, (x, y, w, h))
            energies[i] = energy

        energies = (energies - np.min(energies)) / np.ptp(energies)
        energies[energies < self.minimum_energy] = 0.
        return energies

    def get_pressed_buttons(self, index):
        """
        Returns list of pressed buttons for image in data
        :param index: index of image in data
        :return: list of pressed buttons, color image, thermal image
        """
        color, thermal = self.read_flir(index)
        energy = self.get_energy_count(thermal)
        indexes = np.argwhere(energy > 0.).ravel()
        energies = energy[indexes]
        buttons_indexes = np.argsort(energies)[-4:]
        buttons = indexes[buttons_indexes]
        return buttons, color, thermal

    def draw_reference_lines(self, image, use_ir_positions=True):
        """
        Draws reference lines on image
        :param image: image to draw in
        :param use_ir_positions: use ir positions, else use color positions
        :return: image with lines drawn
        """
        if use_ir_positions:
            for (x, y, w, h) in self.ir_positions:
                image = cv2.rectangle(image, (x, y), (x + w, y + h), color=(0., 0., 1.), thickness=1)
        else:
            for (x, y, w, h) in self.positions:
                image = cv2.rectangle(image, (x, y), (x + w, y + h), color=(0., 0., 1.), thickness=1)

        return image

    def plot_sample(self, color, thermal):
        """
        Sample
        :param color:
        :param thermal:
        :return:
        """
        color = self.draw_reference_lines(color, use_ir_positions=False)
        thermal_lines = thermal.copy()
        thermal_lines = self.draw_reference_lines(thermal_lines)
        mask = self.thresholder(thermal)
        mask_lines = self.draw_reference_lines(mask)

        plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.imshow(color, cmap='gray')
        plt.subplot(1, 2, 1)
        plt.imshow(thermal_lines, cmap='jet')
        plt.subplot(1, 2, 2)
        plt.imshow(mask_lines)


if __name__ == "__main__":
    reader = WarmCodesReader('datos.csv', thresholding='own', intensity_factor=0.5, minimum_energy=0.)

    f = open('digitos.csv', 'w')
    filter = reader.data['digits'] == ''  # all non labeled
    # filter = reader.data['file'] == 'DIGITOS_032'   # only one file

    for index, data in reader.data.loc[filter].iterrows():
        buttons, color, thermal = reader.get_pressed_buttons(index)
        file_name = reader.data.iloc[index]['file']
        file_result = f"{file_name}; {','.join(buttons.astype(str).tolist())}"

        # print and write result to csv
        f.write(f"{file_result}\n")
        print(file_result)

        # plot result for debugging purposes only
        # reader.plot_sample(color, thermal)
        # plt.suptitle(
        #     f"Index: {index}: File: {reader.data.iloc[index]['file']} Label: {reader.data.iloc[index]['digits']} "
        #     f"Prediction: {buttons}")
        # plt.tight_layout()
        # plt.savefig(f"out/{reader.data.iloc[index]['file']}.png")

    f.close()
