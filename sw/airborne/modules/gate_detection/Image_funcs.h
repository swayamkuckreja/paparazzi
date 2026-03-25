#ifndef IMAGE_UTILS
#define IMAGE_UTILS

#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>
#include <string.h>


typedef struct {
    unsigned char ***data;
    int height;
    int width;
    int channels;
} Image;

typedef struct {
    double x;
    double y;
    int valid;
} Point2D;


Image image2array(const char *filename);
void free_image(Image *img);

int **YUV_Filter(Image *img);
void free_filtered_image(int **filtered_image);
void print_image(int **filtered_image, int width, int height);
int **binary_convolution(int **filtered_image, int width, int height);
Point2D centre_of_mass(int **convolved_image, int rows, int cols, int threshold);


#endif 