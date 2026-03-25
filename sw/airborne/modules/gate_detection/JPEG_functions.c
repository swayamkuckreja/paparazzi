#include "Image_funcs.h"

/*
typedef struct {
    unsigned char ***data;
    int height;
    int width;
    int channels;
} Image;
*/

unsigned char ***allocate_3d(int height, int width, int channels) {
    unsigned char ***array = malloc(height * sizeof(unsigned char **));
    if (!array) return NULL;

    unsigned char *data_block = malloc(height * width * channels * sizeof(unsigned char));
    if (!data_block) {
        free(array);
        return NULL;
    }

    for (int y = 0; y < height; y++) {
        array[y] = malloc(width * sizeof(unsigned char *));
        for (int x = 0; x < width; x++) {
            array[y][x] = data_block + (y * width + x) * channels;
        }
    }

    return array;
}


Image image2array(const char *filename){
    // const char *filename = "test.jpg";

    Image img = {0};

    FILE *infile = fopen(filename, "rb");
    if (!infile) {

        fprintf(stderr, "Cannot open file \n");
        return img;
    }


    // Setting up deconpression structures:
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);

    // Read Jpeg Header:
    jpeg_read_header(&cinfo, TRUE);

    // Set output colour space to YCbCr (YUV):
    cinfo.out_color_space = JCS_YCbCr;

    // Start Decompression:
    jpeg_start_decompress(&cinfo);

    int width = cinfo.output_width;
    int height = cinfo.output_height;
    int channels = cinfo.output_components;

    printf("JPEG size: %dx%d, channels=%d (YUV)\n", width, height, channels);

    size_t size = (size_t)width * height * channels;
    unsigned char ***data = allocate_3d(height, width, channels);
    if (!data) {
        fprintf(stderr, "Memory allocation failed\n");
        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        return img;
    }

    // Allocate buffer for one scanline:
    JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray)
                        ((j_common_ptr)&cinfo, JPOOL_IMAGE, width * channels, 1);


    for (int y = 0; y < height; y++) {
        jpeg_read_scanlines(&cinfo, buffer, 1);
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                data[y][x][c] = buffer[0][x * channels + c];
            }
        }
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);


    img.data = data;
    img.height = height;
    img.width = width;
    img.channels = channels;

    return img;

    // Image data stored as img.data[height][width][channel(Y=0)(U=1)(V=2)]
}

void free_image(Image *img) {
    if (!img || !img->data)
        return;

    // The contiguous pixel data block is stored in the first row of the first row-pointer array
    unsigned char *data_block = img->data[0][0];

    // Free each row array
    for (int y = 0; y < img->height; y++) {
        free(img->data[y]);  // free row pointers for each row
    }

    // Free the array of row pointers
    free(img->data);

    // Free the contiguous pixel data
    free(data_block);

    // Reset struct
    img->data = NULL;
    img->height = 0;
    img->width = 0;
    img->channels = 0;
}
