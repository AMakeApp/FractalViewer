import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import matplotlib.cm as cm
import colorcet as cc
from matplotlib.colors import Normalize
from PIL import Image

# np.set_printoptions(threshold=np.nan)


class Fractal:
    MANDELBROT = 0
    JULIA = 1


""" CUDA C functions
    - mandelbrot: calculating mandelbrot set
    :param matrix (float): start x, start y, image size(length), precision
    :param result (float): computed fractal image array pointer
"""

cu = SourceModule("""
    __global__ void calcMandelbrot(bool *isJulia, double *c, double *matrix, int *result) {
        int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
        int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
        double a = matrix[0] + (float)idx_x * matrix[2];
        double b = matrix[1] + (float)idx_y * matrix[2];

        double an = a, bn = b;
        double aan = an * an, bbn = bn * bn;

        for (int i = 0; i < 1000; i++) {
            if (isJulia[0]) { // Julia Set
                bn = 2 * an * bn + c[1];
                an = aan - bbn + c[0];
                aan = an * an;
                bbn = bn * bn;
            } else { // Mandelbrot Set
                bn = 2 * an * bn + b;
                an = aan - bbn + a;
                aan = an * an;
                bbn = bn * bn;
            }
            
            if (an * an + bn * bn > 4.0f) {
                result[idx_x * gridDim.y * blockDim.y + idx_y] = i;
                break;
            }
        }
    }
""")


def calcMandelbrot(type, start, size, unit, c=0 + 0j):
    """ Calculating and drawing mandelbrot set and julia set
    :param type: fractal type
    :param start: center point in plane
    :param size: image size(width, height)
    :param unit: gap among pixels
    :param c: julia constant
    :return: image 2D array (format: RGBA)
    """

    # change size to multiple of 16(bigger or equal to prior size). Because of threads per block is 16
    size = ((size + 255) / 256).astype(np.uint32) * 256

    result = np.empty((size[0], size[1]), np.int32)

    func = cu.get_function("calcMandelbrot")
    func(cuda.In(np.array([type == Fractal.JULIA], np.bool)),
         cuda.In(np.array([c.real, c.imag], np.float64)),
         cuda.In(np.array([start[0] - size[0] / 2 * unit,
                           start[1] - size[1] / 2 * unit, unit], np.float64)),
         cuda.Out(result),
         block=(16, 16, 1), grid=(int(size[0] / 16), int(size[1] / 16)))

    # Because in image symmetric transformation occurs between x axis and y axis
    result = np.transpose(result)

    return array2imgarray(result, cc.m_cyclic_wrwbw_40_90_c42_s25)


def array2imgarray(array, cmap):
    """ Convert set array to image array with drawing pyplot
    :param array: set array
    :param cmap: pyplot colormap
    :return: image 2D array(format: PIL Image RGBA)
    """

    imgarray = cm.ScalarMappable(Normalize(vmin=0, vmax=255), cmap).to_rgba(array) * 255
    return Image.fromarray(np.uint8(imgarray))
