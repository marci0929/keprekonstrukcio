import numpy
from algotom.io.loadersaver import load_image
from algotom.util.simulation import make_sinogram
from algotom.rec.reconstruction import fbp_reconstruction
import algotom.io.loadersaver as losa
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import math


def make_angle_list(number_of_projections):
    return numpy.linspace(0, math.pi, number_of_projections + 1)[:-1]


def get_MSE_from_images(img1, img2):
    return mean_absolute_error(img1, img2)


def make_reconstructed_image(projection_number, original_image):
    angles = make_angle_list(projection_number)
    sinogram = make_sinogram(original_image, angles)
    return fbp_reconstruction(sinogram, original_image.shape[0] / 2, angles, apply_log=False, gpu=False)


def save_image(path, image):
    losa.save_image(path, image)


def print_progress(progress):
    if progress % 5 == 0:
        print("Progress: " + str(progress))


def test_reconstruction():
    original_image = load_image("./sample_pictures/batman_bin_hires.png")
    test_projections = [180, 160, 140, 120, 100, 80, 60, 40, 30, 20, 10, 5]
    MSE_list = []
    progress = 1

    for projection_count in test_projections:
        reconstructed = make_reconstructed_image(projection_count, original_image)
        save_image("./reconstructed/recon_image_" + str(projection_count) + ".png", reconstructed)
        MSE_list.append(get_MSE_from_images(original_image, reconstructed))

        print_progress(progress)
        progress += 1

    plt.plot(test_projections, MSE_list)
    plt.show()


test_reconstruction()
