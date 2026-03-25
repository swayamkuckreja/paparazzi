#include "Image_funcs.h"

int **YUV_Filter(Image *img){
    

    int width = img->width;
    int height = img->height;
    int channels = img->channels;

    int x, y, c;

    int **filtered_image, *values;
    filtered_image = malloc(height * sizeof(int *));
    values = malloc(height*width*sizeof(int));
    for(y=0;y<height;y++)
    {
        filtered_image[y]=values+(y*width);
    }


    for (y=0;y < height;y++)
    {
        for (x=0; x < width; x++)
        {
            if(img->data[y][x][0]>50 && img->data[y][x][0]<150 && img->data[y][x][1]>150 && img->data[y][x][1]<250 &&
            img->data[y][x][2]>50 && img->data[y][x][2]<150)
            {
                filtered_image[y][x]=1;
            }
            else
            {
                filtered_image[y][x]=0;
            }
        }
    }

    return filtered_image;
}

void free_filtered_image(int **filtered_image)
{
    free(filtered_image[0]);
    free(filtered_image);
}

void print_image(int **filtered_image, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Print 1 or 0, add a space for readability
            printf(filtered_image[y][x] ? "██" : "  ");
        }
        printf("\n"); // new row
    }
}

int **binary_convolution(int **filtered_image, int width, int height) {
    int **convolved_image, *values;
    convolved_image = malloc(height * sizeof(int *));
    values = malloc(height*width*sizeof(int));
    for(int y=0;y<height;y++)
    {
        convolved_image[y]=values+(y*width);
    }

    //kernel = ones(13,1)

    for (int y = 6; y < height - 6; y++) {
        for (int x = 0; x < width - 0; x++) {
            if (filtered_image[y][x] == 1 && filtered_image[y-1][x] == 1 && filtered_image[y-2][x] == 1 && filtered_image[y-3][x] == 1 && filtered_image[y-4][x] == 1 && filtered_image[y-5][x] == 1 && filtered_image[y-6][x] == 1 &&
                filtered_image[y+1][x] == 1 && filtered_image[y+2][x] == 1 && filtered_image[y+3][x] == 1 && filtered_image[y+4][x] == 1 && filtered_image[y+5][x] == 1 && filtered_image[y+6][x] == 1)
                {
                convolved_image[y][x] = 1; 
            }
            else {
                convolved_image[y][x] = 0; 
            }
        }
    }

    return convolved_image;
}


Point2D centre_of_mass(int **matrix, int rows, int cols, int threshold) {
    Point2D result = {0.0, 0.0, 0};
    double sum_x = 0.0, sum_y = 0.0;
    int count = 0;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (matrix[r][c] == 1) {
                sum_x += c;
                sum_y += r;
                count++;
            }
        }
    }

    if (count < threshold) return result;

    result.x     = sum_x / count;
    result.y     = sum_y / count;
    result.valid = 1;
    return result;
}