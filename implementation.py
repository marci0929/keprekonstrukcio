import math
import numpy
from algotom.io.loadersaver import load_image
from algotom.util.simulation import make_sinogram
from algotom.rec.reconstruction import fbp_reconstruction
import algotom.io.loadersaver as losa


def make_angle_list(number_of_projections):
    return numpy.linspace(0, math.pi, number_of_projections)

def run_workflow():
    angles = make_angle_list(300)
    image = load_image("./sample_pictures/batman_bin_hires.png")
    sinogram = make_sinogram(image, angles)
    new_image = fbp_reconstruction(sinogram, image.shape[0]/2, angles, apply_log=False, gpu=False)
    losa.save_image("./new_image.png", new_image)


run_workflow()