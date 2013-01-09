

#define STRIDE(strides, dim, type) (strides[(dim)]/sizeof(type))

enum {X0, Y0, X1, Y1, R, G, B};

void decode_loop_ext(int *data, int *data_strides, int *data_dims,
        int *decoded, int *decoded_strides, int *decoded_dims)
{
    int data_s0 = STRIDE(data_strides, 0, int);
    int data_s1 = STRIDE(data_strides, 1, int);
    int decoded_s0 = STRIDE(decoded_strides, 0, int);
    int decoded_s1 = STRIDE(decoded_strides, 1, int);
    int decoded_s2 = STRIDE(decoded_strides, 2, int);

    int row;
    int x0, x1, y0, y1, r, g, b;
    int i, j;
    int data_pos, decoded_pos;

    // printf("data_dims = %d, %d\n", data_dims[0], data_dims[1]);
    // printf("decoded_dims = %d, %d, %d\n", decoded_dims[0], decoded_dims[1], decoded_dims[2]);

    for (row = 0; row < data_dims[0]; ++row) {
        data_pos = row * data_s0;
        /* get rectangle data */
        x0 = data[data_pos + X0 * data_s1];
        y0 = data[data_pos + Y0 * data_s1];
        x1 = data[data_pos + X1 * data_s1];
        y1 = data[data_pos + Y1 * data_s1];
        r  = data[data_pos + R  * data_s1];
        g  = data[data_pos + G  * data_s1];
        b  = data[data_pos + B  * data_s1];
        /* 'paint' rectangle onto surface */
        for (i = y0; i < y1; ++i) {
            for (j = x0; j < x1; ++j) {
                decoded_pos = i*decoded_s0 + j*decoded_s1;
                decoded[decoded_pos                 ] += r;
                decoded[decoded_pos + 1 * decoded_s2] += g;
                decoded[decoded_pos + 2 * decoded_s2] += b;
            }
        }
    }
}
