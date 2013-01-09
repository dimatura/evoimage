
all: decode_loop_ext.so fitness.so

decode_loop_ext.so: decode_loop_ext.c
	gcc -Wall -O2 -ffast-math -o decode_loop_ext.so -shared decode_loop_ext.c

fitness.so: fitness.c
	gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I /usr/include/python2.6/ -o fitness.so fitness.c

fitness.c: fitness.pyx
	cython fitness.pyx
