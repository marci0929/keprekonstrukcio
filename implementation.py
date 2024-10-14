import numpy as np
from algotom.util.simulation import make_sinogram
from algotom.rec.reconstruction import fbp_reconstruction
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import math
import cv2
from collections import Counter


def make_angle_list(number_of_projections):
    return np.linspace(0, math.pi, number_of_projections + 1)[:-1]


def get_mse_from_images(orig, rec):
    return np.count_nonzero(rec - orig) / np.count_nonzero(orig)


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
    mae_list = []
    progress = 1

    for projection_count in test_projections:
        reconstructed = make_reconstructed_image(image, make_angle_list(projection_count))
        save_image("./reconstructed/recon_image_" + str(projection_count) + ".png", reconstructed)
        mae_list.append(get_mse_from_images(image, reconstructed))

        print_progress(progress)
        progress += 1

    plt.plot(test_projections, mae_list)
    plt.show()


def add_trivial_angles(angle_list):
    angle_list.append(0.0)
    angle_list.append(math.pi / 2)


def get_counter_from_vector(angle_list):
    deduplicated = Counter(angle_list)
    return deduplicated


def add_rounded_angle(angle_list, new_angle):
    rounded_angle = round(new_angle, ndigits=4)
    # if not any(angle + 0.01 >= new_angle >= angle - 0.01 for angle in angle_list):
    #     angle_list.append(rounded_angle)
    angle_list.append(rounded_angle)


def get_noisiness_of_signal(sinogram_row):
    noise_value = 0
    for value_idx in range(1, len(sinogram_row)):
        noise_value += abs(sinogram_row[value_idx] - sinogram_row[value_idx - 1])

    return noise_value


def get_noisiness_of_sinogram(sinogram):
    row_count = 0
    row_noise = {}
    for sinogram_row in sinogram:
        row_noise[row_count] = get_noisiness_of_signal(sinogram_row)
        row_count += 1

    return row_noise


def get_improved_sinogram_based_on_noise(sinogram, sinogram_variance):
    # Get the 50 highest noise sinogram rows
    sorted_row_variances = {key: sinogram_variance[key] for key in
                            sorted(sinogram_variance, key=sinogram_variance.get, reverse=True)[:50]}
    new_sinogram = np.array(sinogram[0])
    new_angles = []

    # Append the rows of the original sinogram which has the most noise
    for angle in sorted_row_variances:
        new_sinogram = np.vstack((sinogram[angle], new_sinogram))
        new_angles.insert(0, math.radians(angle))

    # Delete the first dummy row
    new_sinogram = np.delete(new_sinogram, len(new_sinogram) - 1, 0)
    return new_sinogram, new_angles


def noisiness_based_reconstruction(image):
    sinogram = make_sinogram(image, make_angle_list(360))
    sinogram_variance = get_noisiness_of_sinogram(sinogram)
    improved_sinogram, reduced_angles = get_improved_sinogram_based_on_noise(sinogram, sinogram_variance)
    fbp_reconstructed = fbp_reconstruction(improved_sinogram, image.shape[0] / 2, reduced_angles, apply_log=False,
                                           gpu=False)
    return cv2.threshold(fbp_reconstructed, 127, 255, cv2.THRESH_BINARY)[1]


# def test_optimization_with_noisiness(image):
#     reconstructed_img = noisiness_based_reconstruction(image)
#     save_image("./reconstructed/recon_image_noisy_improved.png", reconstructed_img)
#
#     angle_list = make_angle_list(50)
#     reconstructed_img = make_reconstructed_image(image, angle_list)
#     save_image("./reconstructed/recon_image_noisy_reference.png", reconstructed_img)

def test_optimized_reconstruction(image):
    start_projection_count = 50  # number of projections
    projection_list_number = 100  # Number of generated random projections
    max_error_limit = 1  # Error limit, which needs to be achieved
    optimization_step_size = 1.74532925e-5  # 0.001 degree in radians

    continue_optimization = True
    best_projection_error = -1
    best_projection = []
    number_of_projections = start_projection_count

    while continue_optimization:
        random_projections = np.random.uniform(0, math.pi, (number_of_projections, projection_list_number))
        print("Testing " + str(number_of_projections) + " number of projections.")
        print("Current best error: " + str(best_projection_error))

        # Testing the random projections, and selecting the best
        for projection in random_projections:
            error = get_mse_from_images(image, make_reconstructed_image(image, projection))

            if best_projection_error == -1:
                # Handle first reconstruction
                best_projection_error = error
                best_projection = projection
                continue

            if error < best_projection_error:
                best_projection_error = error
                best_projection = projection

        # Here we found the best projection from the random list
        print("Best random projection error: " + str(best_projection_error))
        error_improvement = 1
        best_projection_iter = best_projection
        index_of_optimized_angle = 0
        improved_overall_error = best_projection_error
        tried_angles = 0

        # Greedy algorithm to improve the projection
        print("Greedy optimization in progress...")
        while error_improvement > 0 and tried_angles != len(best_projection_iter):
            print(".", end = " ")
            best_projection_iter[index_of_optimized_angle] += optimization_step_size
            error = get_mse_from_images(image, make_reconstructed_image(image, best_projection_iter))

            if improved_overall_error < best_projection_error and improved_overall_error < max_error_limit:
                # We found a projection which is better than the previous best, and is under the limit
                break

            # Optimization is successful
            if error < improved_overall_error:
                error_improvement = improved_overall_error - error
                improved_overall_error = error
                tried_angles = 0
                continue

            # If addition didn't work, try to substract
            best_projection_iter[index_of_optimized_angle] -= 2 * optimization_step_size
            error = get_mse_from_images(image, make_reconstructed_image(image, best_projection_iter))

            # Optimization is successful
            if error < improved_overall_error:
                error_improvement = improved_overall_error - error
                improved_overall_error = error
                tried_angles = 0
                continue

            tried_angles += 1
            # If we tried to improve all the angles, start again
            if index_of_optimized_angle == len(best_projection_iter) - 1 :
                index_of_optimized_angle = 0
            else:
                index_of_optimized_angle += 1

        print("Best optimized projection error: " + str(improved_overall_error))

        # In this case, the starting number of projections is not good enough
        if improved_overall_error > max_error_limit and number_of_projections == start_projection_count:
            raise Exception(
                "Reconstruction error is bigger than the limit with starting " + str(
                    number_of_projections) + "projections. Please increase the number of the starting projection count!")

        # Optimization successful, try to decrease number of projections
        if improved_overall_error < best_projection_error and improved_overall_error < max_error_limit:
            best_projection_error = improved_overall_error
            best_projection = best_projection_iter
            number_of_projections -= 1
        else:
            # Optimization couldn't find a better reconstruction, we give back the previous projection that was under the error limit
            continue_optimization = False

    print("Least projection achieved is " + number_of_projections)
    save_image("./reconstructed/opt_reconstruction_" + number_of_projections + ".png",
               make_reconstructed_image(image, best_projection))


original_image = cv2.imread("./sample_pictures/sb_hires_inv.png", flags=cv2.IMREAD_GRAYSCALE)
# test_reconstruction(original_image)
test_optimized_reconstruction(original_image)
# test_optimization_with_noisiness(original_image)
