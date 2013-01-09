import numpy as np
cimport numpy as np

cdef enum:
    X0, Y0, X1, Y1, R, G, B

ctypedef np.uint_t uint

cimport cython
@cython.boundscheck(False)
def cy_decode_data(np.ndarray[np.int_t, ndim = 2] scaled,
        np.ndarray[np.int_t, ndim = 3] decoded):
    cdef uint row, i, j
    cdef uint scaled_rows = scaled.shape[0]
    cdef int x0, x1, y0, y1, r, g, b
    for row in range(scaled_rows):
        # without the <uint> gcc barfs
        x0 = scaled[row, <uint> X0]
        y0 = scaled[row, <uint> Y0]
        x1 = scaled[row, <uint> X1]
        y1 = scaled[row, <uint> Y1]
        r  = scaled[row, <uint> R ]
        g  = scaled[row, <uint> G ]
        b  = scaled[row, <uint> B ]
        for i in range(y0, y1):
            for j in range(x0, x1):
                # without the <uint> cython tests for 0 < 0...
                decoded[ i, j, <uint> 0] += r
                decoded[ i, j, <uint> 1] += g
                decoded[ i, j, <uint> 2] += b
