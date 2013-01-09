import numpy as np
cimport numpy as np
cimport cython

cdef extern from "math.h":
    float fabs(float x)

cdef enum:
    X0, Y0, X1, Y1, R, G, B, A

ctypedef np.uint_t uint

@cython.boundscheck(False)
def decode_rectangles(np.ndarray[np.double_t, ndim = 2] rectangles,
        np.ndarray[np.int_t, ndim = 3] decoded):
    cdef uint ix, i, j
    cdef uint num_rectangles = rectangles.shape[0]
    cdef int x0, x1, y0, y1, r, g, b
    cdef double alpha
    cdef int tmp
    cdef uint img_h = decoded.shape[0]
    cdef uint img_w = decoded.shape[1]

    # clear decoded
    decoded.fill(255)

    for ix in range(num_rectangles):
        # without the <uint> gcc barfs
        alpha = rectangles[ix, <uint> A]
        # lower clip alpha
        alpha = 0.1 if alpha < 0.1 else alpha
        alpha = 0.8 if alpha > 0.8 else alpha
        x0 = <int> (rectangles[ix, <uint> X0] * img_w)
        y0 = <int> (rectangles[ix, <uint> Y0] * img_h)
        x1 = <int> (rectangles[ix, <uint> X1] * img_w)
        y1 = <int> (rectangles[ix, <uint> Y1] * img_h)
        r  = <int> (rectangles[ix, <uint> R ] * alpha * 255)
        g  = <int> (rectangles[ix, <uint> G ] * alpha * 255)
        b  = <int> (rectangles[ix, <uint> B ] * alpha * 255)

        # ensure x0 < x1
        if x1 < x0:
            tmp = x1
            x1 = x0
            x0 = tmp
        if y1 < y0:
            tmp = y1
            y1 = y0
            y0 = tmp

        # paint rectangles
        for i in range(y0, y1):
            for j in range(x0, x1):
                # without the <uint> cython tests for 0 < 0...
                # and using 0U does nothing
                decoded[ i, j, <uint> 0] = <np.int_t>(decoded[ i, j, <uint> 0]*(1.0 - alpha))
                decoded[ i, j, <uint> 0] += r
                decoded[ i, j, <uint> 1] = <np.int_t>(decoded[ i, j, <uint> 1]*(1.0 - alpha))
                decoded[ i, j, <uint> 1] += g
                decoded[ i, j, <uint> 2] = <np.int_t>(decoded[ i, j, <uint> 2]*(1.0 - alpha))
                decoded[ i, j, <uint> 2] += b


@cython.boundscheck(False)
def img_diff(np.ndarray[np.uint8_t, ndim = 3] target_img,
             np.ndarray[np.int_t, ndim = 3] decoded):
    cdef uint i, j
    cdef uint img_h = decoded.shape[0]
    cdef uint img_w = decoded.shape[1]
    cdef float error = 0.

    for i in range(img_h):
        for j in range(img_w):
            error += fabs(target_img[i, j, <uint>0] - decoded[i, j, <uint>0])
            error += fabs(target_img[i, j, <uint>1] - decoded[i, j, <uint>1])
            error += fabs(target_img[i, j, <uint>2] - decoded[i, j, <uint>2])
    return error
