#ifndef SFM_TYPES__H
#define SFM_TYPES__H

struct xyz_coords_t
{
    unsigned char num_of_points;
    double *x;
    double *y;
    double *z;
};

// either of x y or z can be NULL so it would not be copied to allocated object
struct xyz_coords_t * alloc_xyz_coords(unsigned char num_of_points = 0, double *x = NULL, double *y = NULL, double *z = NULL);
void free_xyz_coords(struct xyz_coords_t *t);

#endif //SFM_TYPES__H
