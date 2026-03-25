#include "Image_funcs.h"
#include <stdio.h>

int main() {
    Image img = image2array("test.jpg");
    if (!img.data) {
        fprintf(stderr, "Failed to load image\n");
        return 1;
    }

    printf("Loaded image: %dx%d, channels=%d\n", img.width, img.height, img.channels);


    int** filtered_image = YUV_Filter(&img);
    int** convolved_image = binary_convolution(filtered_image, img.width, img.height);
    // print_image(filtered_image, img.width, img.height);
    // print_image(convolved_image, img.width, img.height);
    Point2D centre_of_gate = centre_of_mass(convolved_image, img.height, img.width, 10);
    if (centre_of_gate.valid) {
        printf("Centre of gate: (%.2f, %.2f)\n", centre_of_gate.x, centre_of_gate.y);
    }

    free_filtered_image(filtered_image);
    free_filtered_image(convolved_image);
    free_image(&img);
    return 0;
}


void what_to_do() {
    if (!obstacle_detected) {
        if (gate) {
            if (gate_x <= 0.4*image_width) {
                turn_left();
            }
            else if (gate_x >= 0.6*image_width) {
                turn_right();
            } else {
                go_forward();
            }
        }
    }

}