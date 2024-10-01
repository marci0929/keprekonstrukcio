import numpy
import numpy as np
from algotom.io.loadersaver import load_image
from algotom.util.simulation import make_sinogram
from algotom.rec.reconstruction import fbp_reconstruction
import algotom.io.loadersaver as losa
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import math
import cv2
from collections import Counter


def make_angle_list(number_of_projections):
    return numpy.linspace(0, math.pi, number_of_projections + 1)[:-1]


def get_MSE_from_images(img1, img2):
    return mean_absolute_error(img1, img2)


def make_reconstructed_image(image, projection_angles):
    sinogram = make_sinogram(image, projection_angles)
    fbp_reconstructed = fbp_reconstruction(sinogram, image.shape[0] / 2, projection_angles, apply_log=False, gpu=False)
    return cv2.threshold(fbp_reconstructed, 127, 255, cv2.THRESH_BINARY)[1]


def save_image(path, image):
    cv2.imwrite(path, image)


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


def get_normalized_angle_from_2_coordinates(coord1, coord2):
    delta_x = abs(coord2[0] - coord1[0])
    delta_y = abs(coord2[1] - coord1[1])
    return math.atan2(delta_x, delta_y)


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

    while x_search_offset != image_size:
        get_search_box_values(image, values_list, x_search_offset, y_search_offset)
        x_search_offset += 1
        if x_search_offset != image_size:
            get_search_box_values(image, values_list, x_search_offset, y_search_offset)

        x_search_offset -= 1
        y_search_offset += 1

        if y_search_offset != image_size:
            get_search_box_values(image, values_list, x_search_offset, y_search_offset)

        x_search_offset += 1

    return values_list


def test_reconstruction(image):
    test_projections = [180, 160, 140, 120, 100, 80, 60, 40, 30, 20, 10, 5]
    MAE_list = []
    progress = 1

    for projection_count in test_projections:
        reconstructed = make_reconstructed_image(image, make_angle_list(projection_count))
        save_image("./reconstructed/recon_image_" + str(projection_count) + ".png", reconstructed)
        MAE_list.append(get_MSE_from_images(image, reconstructed))

        print_progress(progress)
        progress += 1

    plt.plot(test_projections, MAE_list)
    plt.show()


def add_trivial_angles(angle_list):
    angle_list.append(0.0)
    angle_list.append(math.pi / 2)


def get_counter_from_vector(angle_list):
    deduplicated = Counter(angle_list)
    return deduplicated


def add_rounded_angle(angle_list, new_angle):
    rounded_angle = round(new_angle, ndigits=2)
    if not any(angle + 0.01 >= new_angle >= angle - 0.01 for angle in angle_list):
        angle_list.append(rounded_angle)


def test_optimized_reconstruction(image):
    components = get_list_of_switching_components(image)

    angles = []
    for comp in components:
        normalized_angle_1 = get_normalized_angle_from_2_coordinates(comp[0], comp[1])
        # Get the other 2 coordinates of the switching component as: (x, y + y_offset) and (x + x_offset, y)
        normalized_angle_2 = math.pi - normalized_angle_1

        add_rounded_angle(angles, normalized_angle_1)
        add_rounded_angle(angles, normalized_angle_2)

    counter_set = get_counter_from_vector(angles)
    most_common = counter_set.most_common(20)

    reduced_angles = [angle[0] for angle in most_common]

    add_trivial_angles(reduced_angles)

    print("total projections: " + str(len(reduced_angles)))

    reconstructed_img = make_reconstructed_image(image, reduced_angles)
    save_image("./reconstructed/recon_image_improved.png", reconstructed_img)

    angle_list = make_angle_list(16)
    reconstructed_img = make_reconstructed_image(image, angle_list)
    save_image("./reconstructed/recon_image_reference.png", reconstructed_img)


original_image = cv2.imread("./sample_pictures/batman_bin_lowres.png", flags=cv2.IMREAD_GRAYSCALE)
# test_reconstruction(original_image)
test_optimized_reconstruction(original_image)
