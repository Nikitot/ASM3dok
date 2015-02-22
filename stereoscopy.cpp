#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "stereoscopy.h"

#define MODULE_NAME "STEREOSCOPY"

void free_rotation_matrix(double **R)
{
    int i;
    for(i = 0; i < 3; i++)
    {
        free(R[i]);
    }
    free(R);
}

double ** alloc_rotation_matrix(double alpha, double betta, double gamma)
{
    int i;
    double **r;

    r = (double**)calloc(3 * sizeof(double*), 1);
    if(r == NULL)
    {
        printf("calloc failed on error %s", strerror(errno));
        return NULL;
    }
    for(i = 0; i < 3; i++)
    {
        r[i] = (double*)calloc(3 * sizeof(double), 1);
        if(r[i] == NULL)
        {
			printf("calloc failed on error %s", strerror(errno));
            free_rotation_matrix(r);
            return NULL;
        }
    }

    r[0][0] = cos(betta) * cos(gamma);
    r[0][1] = sin(alpha) * sin(betta) * cos(gamma) - cos(alpha) * sin(gamma);
    r[0][2] = cos(alpha) * sin(betta) * cos(gamma) + sin(alpha) * sin(gamma);

    r[1][0] = sin(gamma) * cos(betta);
    r[1][1] = sin(alpha) * sin(betta) * sin(gamma) + cos(alpha) * cos(gamma);
    r[1][2] = cos(alpha) * sin(betta) * sin(gamma) - sin(alpha) * cos(gamma);

    r[2][0] = -sin(betta);
    r[2][1] = sin(alpha) * cos(betta);
    r[2][2] = cos(alpha) * cos(betta);
    return r;
}

struct xyz_coords_t *get_3d_object_coords(struct xyz_coords_t *coords_1, double angle_1,
                                          struct xyz_coords_t *coords_2, double angle_2)
{
    int i;
    double **rot_mat_2;
    struct xyz_coords_t *res;

    if(coords_1 == NULL || coords_2 == NULL)
    {
        printf("Invalid parameters. coords_1 = %p, coords_2 = %p", coords_1, coords_2);
        return NULL;
    }

    if(coords_1->num_of_points == 0 || coords_2->num_of_points == 0
            || (coords_1->num_of_points != coords_2->num_of_points))
    {
		printf("Invalid num of points: %d %d %d", coords_1->num_of_points, coords_2->num_of_points);
        return NULL;
    }

    res = (struct xyz_coords_t*)calloc(sizeof(struct xyz_coords_t), 1);
    if(res == NULL)
    {
		printf("calloc failed on error %s", strerror(errno));
        return NULL;
    }

    if(NULL == (rot_mat_2 = alloc_rotation_matrix(0, angle_2, 0)))
    {
		printf("getRotationMatrix failed");
        return NULL;
    }

    if(NULL == (res = alloc_xyz_coords(coords_1->num_of_points)))
    {
		printf("alloc_xyz_coords failed for res coords");
        free_rotation_matrix(rot_mat_2);
        return NULL;
    }

    for(i = 0; i < res->num_of_points; i++)
    {
        res->x[i] = rot_mat_2[1][1] * coords_2->x[i] + coords_2->y[i] * rot_mat_2[1][2] + rot_mat_2[1][3];
        res->y[i] = rot_mat_2[2][1] * coords_2->x[i] + coords_2->y[i] * rot_mat_2[2][2] + rot_mat_2[2][3];
        res->z[i] = (coords_1->x[i] - coords_2->x[i]) / sin(angle_1 - angle_2);
    }

    free_rotation_matrix(rot_mat_2);

    return res;
}
