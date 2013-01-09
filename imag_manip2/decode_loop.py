
import ctypes
import numpy as np

lib = np.ctypeslib.load_library('decode_loop_ext', '.')

lib.decode_loop_ext.restype = None
lib.decode_loop_ext.argtypes = [\
        np.ctypeslib.ndpointer(np.int, flags = 'aligned'),
        ctypes.POINTER(np.ctypeslib.c_intp), # strides
        ctypes.POINTER(np.ctypeslib.c_intp), # dims
        np.ctypeslib.ndpointer(np.int, flags = 'aligned, writeable'),
        ctypes.POINTER(np.ctypeslib.c_intp), # strides
        ctypes.POINTER(np.ctypeslib.c_intp), # dims
        ]

def decode_loop(data, decoded):
    lib.decode_loop_ext(\
            data, data.ctypes.strides, data.ctypes.shape,
            decoded, decoded.ctypes.strides, decoded.ctypes.shape)
