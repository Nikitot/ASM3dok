#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include "sfm_types.h"

#define MODULE_NAME "SFM"

struct xyz_coords_t * alloc_xyz_coords(unsigned char num_of_points, double *x, double *y, double *z)
{
    xyz_coords_t *t = (struct xyz_coords_t*)calloc(sizeof(struct xyz_coords_t), 1);
    if(t == NULL)
    {
        printf("calloc failed on error %s", strerror(errno));
        return NULL;
    }

    if(num_of_points != 0)
    {
        t->num_of_points = num_of_points;

        t->x = (double*)calloc(num_of_points * sizeof(double), 1);
        if(t->x == NULL)
        {
            printf("calloc failed on error %s", strerror(errno));
            free(t);
            return NULL;
        }

        if(x != NULL)
        {
            memcpy(t->x, x, num_of_points * sizeof(double));
        }

        t->y = (double*)calloc(num_of_points * sizeof(double), 1);
        if(t->y == NULL)
        {
			printf("calloc failed on error %s", strerror(errno));
            free(t->x);
            free(t);
            return NULL;
        }

        if(y != NULL)
        {
            memcpy(t->y, y, num_of_points * sizeof(double));
        }

        t->z = (double*)calloc(num_of_points * sizeof(double), 1);
        if(t->z == NULL)
        {
			printf("calloc failed on error %s", strerror(errno));
            free(t->y);
            free(t->x);
            free(t);
            return NULL;
        }

        if(z != NULL)
        {
            memcpy(t->z, z, num_of_points * sizeof(double));
        }
    }

    return t;
}

void free_xyz_coords(struct xyz_coords_t *t)
{
    if(t != NULL)
    {
        free(t->x);
        free(t->y);
        free(t->z);
        free(t);
    }
}
