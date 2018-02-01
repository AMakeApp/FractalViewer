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
    __global__ void mandelbrot(double *matrix, int *result) {
        double a = matrix[0] + (float)blockIdx.x * matrix[2];
        double b = matrix[1] + (float)blockIdx.y * matrix[2];

        double an = a, bn = b;

        for (int i = 0; i < 250; i++) {
            double tmp = an;
            an = an * an - bn * bn + a;
            bn = 2 * tmp * bn + b;

            if (an * an + bn * bn > 4.0f) {
                result[blockIdx.x * gridDim.y + blockIdx.y] = i;
                break;
            }
        }
    }
    
    __global__ void julia(double *matrix, double *c, int *result) {
        double a = matrix[0] + (float)blockIdx.x * matrix[2];
        double b = matrix[1] + (float)blockIdx.y * matrix[2];
        
        double an = a, bn = b;
        
        for (int i = 0; i < 250; i++) {
            double tmp = an;
            an = an * an - bn * bn + c[0];
            bn = 2 * tmp * bn + c[1];
            
            if (an * an + bn * bn > 4.0f) {
                result[blockIdx.x * gridDim.y + blockIdx.y] = i;
                break;
            }
        }
    }
""")


def drawmandelbrot(start, size, unit):
    """ Calculating and drawing mandelbrot set
    :param start: left-bottom point in plane
    :param size: image size(width, height)
    :param unit: gap among pixels
    :return: image 2D array (format: RGBA)
    """

    size = size.astype(np.uint32, copy=False)

    matrix = np.array([start[0] - size[0] / 2 * unit, start[1] - size[1] / 2 * unit, unit], np.float64)
    result = np.empty((size[0], size[1]), np.int32)
    matrix_gpu = cuda.mem_alloc(matrix.nbytes)
    result_gpu = cuda.mem_alloc(result.nbytes)

    cuda.memcpy_htod(matrix_gpu, matrix)

    func = cu.get_function("mandelbrot")
    func(matrix_gpu, result_gpu, block=(1, 1, 1), grid=(int(size[0]), int(size[1])))

    cuda.memcpy_dtoh(result, result_gpu)

    # Because in image symmetric transformation occurs between x axis and y axis
    result = np.transpose(result)

    return array2imgarray(result, cc.m_cyclic_wrwbw_40_90_c42_s25)


def drawjulia(start, size, unit, c):
    """ Calculating and drawing julia set
    :param start: left-bottom point in plane
    :param size: image size(width, height)
    :param unit: gap among pixels
    :param c: constant for julia set
    :return: image 2D array (format: RGBA)
    """

    size = size.astype(np.uint32, copy=False)

    matrix = np.array([start[0] - size[0] / 2 * unit, start[1] - size[1] / 2 * unit, unit], np.float64)
    c = np.array([c.real, c.imag], np.float64)
    result = np.empty((size[0], size[1]), np.int32)
    matrix_gpu = cuda.mem_alloc(matrix.nbytes)
    c_gpu = cuda.mem_alloc(c.nbytes)
    result_gpu = cuda.mem_alloc(result.nbytes)

    cuda.memcpy_htod(matrix_gpu, matrix)
    cuda.memcpy_htod(c_gpu, c)

    func = cu.get_function("julia")
    func(matrix_gpu, c_gpu, result_gpu, block=(1, 1, 1), grid=(int(size[0]), int(size[1])))

    cuda.memcpy_dtoh(result, result_gpu)

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
