import numpy
from algotom.io.loadersaver import load_image
from algotom.util.simulation import make_sinogram
from algotom.rec.reconstruction import fbp_reconstruction
import algotom.io.loadersaver as losa
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import math
import cv2


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


# Check if the corners of the search box is a switching component
# The box is defined as the following:
# a ---- b
# |      |
# c ---- d
def check_if_switching_component(a, b, c, d):
    return a == d and c == b and a != b


def get_search_box_values(image, values_list, x_offset, y_offset):
    image_size = image.shape[0]

    for y in range(image_size - y_offset):
        for x in range(image_size - x_offset):
            is_switching_component = check_if_switching_component(image[y][x], image[y][x + x_offset],
                                                                  image[y + y_offset][x],
                                                                  image[y + y_offset][x + x_offset])
            if is_switching_component:
                values_list.append(((x, y), (x + x_offset, y + y_offset)))


def get_list_of_switching_components(image):
    x_search_offset = 1
    y_search_offset = 1

    image_size = image.shape[0]

    # Contains the switching component coordinates in (x, y) form in 4 element tuples as follows:
    # a ---- b
    # |      |
    # c ---- d
    # (a, b, c, d)
    values_list = []

    while x_search_offset != image_size and y_search_offset != image_size:
        get_search_box_values(image, values_list, x_search_offset, y_search_offset)
        x_search_offset += 1
        if x_search_offset != image_size:
            get_search_box_values(image, values_list, x_search_offset, y_search_offset)

        x_search_offset -= 1
        y_search_offset += 1

        if x_search_offset != image_size:
            get_search_box_values(image, values_list, x_search_offset, y_search_offset)

        x_search_offset += 1

    return values_list


def test_reconstruction(image):
    test_projections = [180, 160, 140, 120, 100, 80, 60, 40, 30, 20, 10, 5]
    MSE_list = []
    progress = 1

    for projection_count in test_projections:
        reconstructed = make_reconstructed_image(projection_count, image)
        save_image("./reconstructed/recon_image_" + str(projection_count) + ".png", reconstructed)
        MSE_list.append(get_MSE_from_images(image, reconstructed))

        print_progress(progress)
        progress += 1

    plt.plot(test_projections, MSE_list)
    plt.show()


original_image = cv2.imread("./sample_pictures/switching_1.png", flags=cv2.IMREAD_GRAYSCALE)
# test_reconstruction(original_image)
components = get_list_of_switching_components(original_image)
print(components)