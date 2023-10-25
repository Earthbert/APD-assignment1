// Author: APD team, except where source was noted

#include "helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#define CONTOUR_CONFIG_COUNT    16
#define FILENAME_MAX_SIZE       50
#define STEP                    8
#define SIGMA                   200
#define RESCALE_X               2048
#define RESCALE_Y               2048

#define CLAMP(v, min, max) if(v < min) { v = min; } else if(v > max) { v = max; }

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

typedef struct {
    int id;
    int num_threads;
    pthread_barrier_t *barrier;
    ppm_image *main_image;
    ppm_image **rescaled_image_p;
    struct {
        unsigned char ***grid_p;
        int sigma;
        int step_x;
        int step_y;
    } sample_grid;
    ppm_image ***contour_map_p;
} thread_args_t;

thread_args_t *init_thread_args(int num_threads, int id, pthread_barrier_t *barrier,
    ppm_image **rescaled_image_p, ppm_image *image,
    unsigned char ***grid_p, int step_x, int step_y, int sigma, ppm_image ***contour_map_p) {

    thread_args_t *args = malloc(sizeof(thread_args_t));

    args->num_threads = num_threads;
    args->id = id;
    args->barrier = barrier;
    args->rescaled_image_p = rescaled_image_p;
    args->main_image = image;
    args->sample_grid.grid_p = grid_p;
    args->sample_grid.step_x = step_x;
    args->sample_grid.step_y = step_y;
    args->sample_grid.sigma = sigma;
    args->contour_map_p = contour_map_p;

    return args;
}

void bicubic_interp_work(int start, int end, ppm_image *new_image, ppm_image *image) {
    uint8_t sample[3];

    // use bicubic interpolation for scaling
    for (int i = start; i < end; i++) {
        for (int j = 0; j < new_image->y; j++) {
            float u = (float)i / (float)(new_image->x - 1);
            float v = (float)j / (float)(new_image->y - 1);
            sample_bicubic(image, u, v, sample);

            new_image->data[i * new_image->y + j].red = sample[0];
            new_image->data[i * new_image->y + j].green = sample[1];
            new_image->data[i * new_image->y + j].blue = sample[2];
        }
    }
}

void sample_grid_work(ppm_image *image, unsigned char **grid, int sigma,
    int step_x, int step_y, int id, int num_threads) {
    int p = image->x / step_x;
    int q = image->y / step_y;

    int p_start = id * (double)p / num_threads;
    int p_end = MIN((id + 1) * (double)p / num_threads, p);

    int q_start = id * (double)q / num_threads;
    int q_end = MIN((id + 1) * (double)q / num_threads, q);

    for (int i = p_start; i < p_end; i++) {
        for (int j = 0; j < q; j++) {
            ppm_pixel curr_pixel = image->data[i * step_x * image->y + j * step_y];

            unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

            if (curr_color > sigma) {
                grid[i][j] = 0;
            } else {
                grid[i][j] = 1;
            }
        }
    }

    // last sample points have no neighbors below / to the right, so we use pixels on the
    // last row / column of the input image for them
    for (int i = p_start; i < p_end; i++) {
        ppm_pixel curr_pixel = image->data[i * step_x * image->y + image->x - 1];

        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > sigma) {
            grid[i][q] = 0;
        } else {
            grid[i][q] = 1;
        }
    }
    for (int j = q_start; j < q_end; j++) {
        ppm_pixel curr_pixel = image->data[(image->x - 1) * image->y + j * step_y];

        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > sigma) {
            grid[p][j] = 0;
        } else {
            grid[p][j] = 1;
        }
    }
}

// Updates a particular section of an image with the corresponding contour pixels.
// Used to create the complete contour image.
void update_image(ppm_image *image, ppm_image *contour, int x, int y) {
    for (int i = 0; i < contour->x; i++) {
        for (int j = 0; j < contour->y; j++) {
            int contour_pixel_index = contour->x * i + j;
            int image_pixel_index = (x + i) * image->y + y + j;

            image->data[image_pixel_index].red = contour->data[contour_pixel_index].red;
            image->data[image_pixel_index].green = contour->data[contour_pixel_index].green;
            image->data[image_pixel_index].blue = contour->data[contour_pixel_index].blue;
        }
    }
}

// Corresponds to step 2 of the marching squares algorithm, which focuses on identifying the
// type of contour which corresponds to each subgrid. It determines the binary value of each
// sample fragment of the original image and replaces the pixels in the original image with
// the pixels of the corresponding contour image accordingly.
void march(ppm_image *image, unsigned char **grid, ppm_image **contour_map, int step_x, int step_y,
    int id, int num_threads) {
    int p = image->x / step_x;
    int q = image->y / step_y;

    int start = id * (double)p / num_threads;
    int end = MIN((id + 1) * (double)p / num_threads, p);

    for (int i = start; i < end; i++) {
        for (int j = 0; j < q; j++) {
            unsigned char k = 8 * grid[i][j] + 4 * grid[i][j + 1] + 2 * grid[i + 1][j + 1] + 1 * grid[i + 1][j];
            update_image(image, contour_map[k], i * step_x, j * step_y);
        }
    }
}

void *thread_func(void *void_args) {
    thread_args_t *args = (thread_args_t *)void_args;

    // Wait for all threads to be ready to start rescaling
    pthread_barrier_wait(args->barrier);

    // Check if rescaling is needed
    if (*(args->rescaled_image_p) != NULL) {
        ppm_image *new_image = *(args->rescaled_image_p);
        int start = args->id * (double)new_image->x / args->num_threads;
        int end = MIN((args->id + 1) * (double)new_image->x / args->num_threads, new_image->x);

        bicubic_interp_work(start, end, new_image, args->main_image);

        args->main_image = new_image;

        // Wait for all threads to finish rescaling
        pthread_barrier_wait(args->barrier);
    }

    // Wait for all threads to be ready to start grid sampling
    pthread_barrier_wait(args->barrier);

    sample_grid_work(args->main_image, *(args->sample_grid.grid_p), args->sample_grid.sigma,
        args->sample_grid.step_x, args->sample_grid.step_y, args->id, args->num_threads);

    // Wait for all threads to finish grid sampling
    pthread_barrier_wait(args->barrier);

    march(args->main_image, *(args->sample_grid.grid_p), *(args->contour_map_p),
        args->sample_grid.step_x, args->sample_grid.step_y, args->id, args->num_threads);

    // Wait for all threads to finish march
    pthread_barrier_wait(args->barrier);

    return NULL;
}

// Creates a map between the binary configuration (e.g. 0110_2) and the corresponding pixels
// that need to be set on the output image. An array is used for this map since the keys are
// binary numbers in 0-15. Contour images are located in the './contours' directory.
ppm_image **init_contour_map() {
    ppm_image **map = (ppm_image **)malloc(CONTOUR_CONFIG_COUNT * sizeof(ppm_image *));
    if (!map) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
        char filename[FILENAME_MAX_SIZE];
        sprintf(filename, "./contours/%d.ppm", i);
        map[i] = read_ppm(filename);
    }

    return map;
}

// Corresponds to step 1 of the marching squares algorithm, which focuses on sampling the image.
// Builds a p x q grid of points with values which can be either 0 or 1, depending on how the
// pixel values compare to the `sigma` reference value. The points are taken at equal distances
// in the original image, based on the `step_x` and `step_y` arguments.
unsigned char **sample_grid(ppm_image *image, int step_x, int step_y, unsigned char ***grid_p, pthread_barrier_t *barrier) {
    int p = image->x / step_x;
    int q = image->y / step_y;

    unsigned char **grid = (unsigned char **)malloc((p + 1) * sizeof(unsigned char *));
    if (!grid) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    for (int i = 0; i <= p; i++) {
        grid[i] = (unsigned char *)malloc((q + 1) * sizeof(unsigned char));
        if (!grid[i]) {
            fprintf(stderr, "Unable to allocate memory\n");
            exit(1);
        }
    }

    *grid_p = grid;

    // Wait for all threads to be ready to start work
    pthread_barrier_wait(barrier);

    // Wait for all threads to finish work
    pthread_barrier_wait(barrier);

    return grid;
}

// Calls `free` method on the utilized resources.
void free_resources(ppm_image *image, ppm_image **contour_map, unsigned char **grid, int step_x,
    thread_args_t **args, int num_threads) {
    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++) {
        free(contour_map[i]->data);
        free(contour_map[i]);
    }
    free(contour_map);

    for (int i = 0; i <= image->x / step_x; i++) {
        free(grid[i]);
    }
    free(grid);

    for (int i = 0; i < num_threads; i++)
    {
        free(args[i]);
    }

    free(image->data);
    free(image);
}

ppm_image *rescale_image(ppm_image *image, ppm_image **rescaled_img,
    pthread_barrier_t *barrier) {
    // we only rescale downwards
    if (image->x <= RESCALE_X && image->y <= RESCALE_Y) {
        pthread_barrier_wait(barrier);
        return image;
    }

    // alloc memory for image
    ppm_image *new_image = (ppm_image *)malloc(sizeof(ppm_image));
    if (!new_image) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }
    new_image->x = RESCALE_X;
    new_image->y = RESCALE_Y;

    new_image->data = (ppm_pixel *)malloc(new_image->x * new_image->y * sizeof(ppm_pixel));
    if (!new_image) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    *rescaled_img = new_image;

    // Wait for all threads to be ready to start
    pthread_barrier_wait(barrier);

    // Wait for all threads to finish work
    pthread_barrier_wait(barrier);

    free(image->data);
    free(image);

    return new_image;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: ./tema1 <in_file> <out_file> <P>\n");
        return 1;
    }

    ppm_image *image = read_ppm(argv[1]);
    int step_x = STEP;
    int step_y = STEP;

    int num_threads = atoi(argv[3]);

    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, num_threads + 1);

    pthread_t threads[num_threads];
    thread_args_t *args[num_threads];

    ppm_image *rescaled_img = NULL;
    ppm_image **contour_map = NULL;

    unsigned char **grid = NULL;

    // Create worker threads
    for (long id = 0; id < num_threads; id++) {
        args[id] = init_thread_args(num_threads, id, &barrier,
            &rescaled_img, image, &grid, step_x, step_y, SIGMA, &contour_map);

        int r = pthread_create(&threads[id], NULL, thread_func, args[id]);

        if (r) {
            printf("Eroare la crearea thread-ului %ld\n", id);
            exit(-1);
        }
    }

    // 0. Initialize contour map
    contour_map = init_contour_map();

    // 1. Rescale the image
    ppm_image *scaled_image = rescale_image(image, &rescaled_img, &barrier);

    // 2. Sample the grid
    grid = sample_grid(scaled_image, step_x, step_y, &grid, &barrier);

    // 3. Wait for march the squares
    pthread_barrier_wait(&barrier);

    // Wait for all worker threads to finish (I know the barrier above is useless)
    for (long id = 0; id < num_threads; id++) {
        void *status;
        int r = pthread_join(threads[id], &status);

        if (r) {
            printf("Eroare la asteptarea thread-ului %ld\n", id);
            exit(-1);
        }
    }

    // 4. Write output
    write_ppm(scaled_image, argv[2]);

    free_resources(scaled_image, contour_map, grid, step_x, args, num_threads);

    return 0;
}
