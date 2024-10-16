import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from algotom.rec.reconstruction import fbp_reconstruction
from algotom.util.simulation import make_sinogram

rng = np.random.default_rng()
greedy_progress = 0


def make_angle_list(number_of_projections):
    return np.linspace(0, math.pi, number_of_projections + 1)[:-1]


def get_error_from_images(orig, rec):
    return np.count_nonzero(rec - orig) / np.count_nonzero(orig)


def make_reconstructed_image(image, projection_angles):
    sinogram = make_sinogram(image, projection_angles)
    fbp_reconstructed = fbp_reconstruction(sinogram, image.shape[0] / 2, projection_angles, apply_log=False, gpu=False)
    return cv2.threshold(fbp_reconstructed, 127, 255, cv2.THRESH_BINARY)[1]


def save_image(path, image):
    cv2.imwrite(path, image)


def print_progress(progress):
    if progress % 2 == 0:
        print("Progress: " + str(progress))


def test_reconstruction(image):
    test_projections = [180, 160, 140, 120, 100, 80, 60, 40, 30, 20, 10, 5]
    mae_list = []
    progress = 1

    for projection_count in test_projections:
        reconstructed = make_reconstructed_image(image, make_angle_list(projection_count))
        save_image("./reconstructed/recon_image_" + str(projection_count) + ".png", reconstructed)
        mae_list.append(get_error_from_images(image, reconstructed))

        print_progress(progress)
        progress += 1

    plt.plot(test_projections, mae_list)
    plt.show()


def test_optimization(image, equal_projection_count):
    equal_angle_reconstructed = make_reconstructed_image(image, make_angle_list(equal_projection_count))
    equal_angles_error = get_error_from_images(image, equal_angle_reconstructed)
    save_image("./reconstructed/equal_reconst_" + str(equal_projection_count) + ".png", equal_angle_reconstructed)
    print("Equal angle reconstruction error is: " + str(equal_angles_error))

    best_rec, best_rec_proj_cnt, best_rec_error = optimized_reconstruction(image, equal_angles_error)

    print("Optimized reconstruction with same error made from " + str(
        best_rec_proj_cnt) + " projections, instead of " + str(
        equal_projection_count) + ", with error " + str(best_rec_error))
    save_image("./reconstructed/opt_recon_" + str(best_rec_proj_cnt) + ".png", best_rec)


def print_greedy_opt_progress(break_size):
    global greedy_progress
    print(".", end=" ")
    greedy_progress += 1
    if greedy_progress == break_size:
        print("\n")
        greedy_progress = 0


def optimized_reconstruction(image, max_error_limit):
    start_projection_count = 20  # number of projections
    number_of_random_projections = 10  # Number of generated random projections
    optimization_step_size = 1e-16  # Number to add and subtract from angles

    number_of_projections = start_projection_count
    continue_optimization = True
    best_overall_projection = []
    best_overall_error = max_error_limit + 1

    while continue_optimization:
        random_projections = make_angle_list(number_of_projections)
        # Add the uniform angle list, it can be better sometimes
        random_projections = np.vstack(
            [random_projections, rng.uniform(0, math.pi,
                                             (number_of_random_projections, number_of_projections))])
        print("Testing " + str(number_of_projections) + " projections.")

        best_current_projection_number_error = \
            (get_error_from_images(image, make_reconstructed_image(image, random_projections[0])))

        # Testing the random projections, and selecting the best one
        for projection in random_projections:
            greedy_best_error = get_error_from_images(image, make_reconstructed_image(image, projection))

            # Here we found the best projection from the random list
            if greedy_best_error <= max_error_limit:
                # We found a projection which is under the limit
                best_overall_projection = projection
                best_current_projection_number_error = greedy_best_error
                break

            # Only to show the correct progress
            global greedy_progress
            greedy_progress = 0
            # -----

            greedy_projection_iter = np.copy(projection)
            index_of_optimized_angle = 0
            tried_angles = 0
            current_step_size = optimization_step_size

            # Greedy algorithm to improve the projection
            while tried_angles != number_of_projections:
                if greedy_best_error <= max_error_limit:
                    # We found a projection which is under the limit
                    best_overall_projection = np.copy(greedy_projection_iter)
                    best_current_projection_number_error = greedy_best_error
                    break

                print_greedy_opt_progress(number_of_projections)

                saved_angle = greedy_projection_iter[index_of_optimized_angle]
                if greedy_projection_iter[index_of_optimized_angle] + current_step_size < math.pi:
                    greedy_projection_iter[index_of_optimized_angle] += current_step_size
                    error_1 = get_error_from_images(image, make_reconstructed_image(image, greedy_projection_iter))

                    # Optimization is successful
                    if error_1 <= greedy_best_error:
                        greedy_best_error = error_1
                        tried_angles = 0
                        current_step_size = optimization_step_size
                        print("!", end=" ")
                        continue

                # If addition didn't work, try to subtract
                if greedy_projection_iter[index_of_optimized_angle] - 2 * current_step_size >= 0:
                    greedy_projection_iter[index_of_optimized_angle] -= 2 * current_step_size
                    error_1 = get_error_from_images(image, make_reconstructed_image(image, greedy_projection_iter))

                    # Optimization is successful
                    if error_1 <= greedy_best_error:
                        greedy_best_error = error_1
                        tried_angles = 0
                        current_step_size = optimization_step_size
                        print("!", end=" ")
                        continue

                # Optimization unsuccessful for this angle, try another one
                greedy_projection_iter[index_of_optimized_angle] = saved_angle
                tried_angles += 1
                index_of_optimized_angle += 1

                # If we tried to improve all the angles, start again
                if index_of_optimized_angle == number_of_projections:
                    index_of_optimized_angle = 0

                # Optimization couldn't find a better reconstruction, try bigger step size if possible, let's see
                if tried_angles == number_of_projections and current_step_size < (math.pi / 2):
                    tried_angles = 0
                    index_of_optimized_angle = 0
                    current_step_size *= 10

            # Greedy found a good projection set
            if best_current_projection_number_error <= max_error_limit:
                # We found a projection which is under the limit
                break
            else:
                print("Optimized random projection best error: " + str(best_current_projection_number_error))

        print("\nBest optimized projection error: " + str(best_current_projection_number_error))

        # In this case, the starting number of projections is not enough even on the start
        if best_current_projection_number_error > max_error_limit and number_of_projections == start_projection_count:
            raise RuntimeError(
                "Reconstruction error is bigger than the limit with starting " + str(
                    number_of_projections) + "projections. Please increase the number"
                                             " of the starting projection count!")

        # Optimization successful, try to decrease number of projections
        if best_current_projection_number_error <= max_error_limit:
            best_overall_error = best_current_projection_number_error
            number_of_projections -= 1
        else:
            # Optimization couldn't find a better reconstruction, we give back the previous projection
            # that was under the error limit
            continue_optimization = False

    print("Least projection achieved is " + str(number_of_projections + 1))
    best_reconstruction = make_reconstructed_image(image, best_overall_projection)
    save_image("./reconstructed/opt_reconstruction_" + str(number_of_projections + 1) + ".png", best_reconstruction)

    return best_reconstruction, number_of_projections + 1, best_overall_error


original_image = cv2.imread("./sample_pictures/batman_bin_lowres.png", flags=cv2.IMREAD_GRAYSCALE)

# test_reconstruction(original_image)
# optimized_reconstruction(original_image, 0.35)
test_optimization(original_image, 20)
