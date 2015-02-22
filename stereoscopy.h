#ifndef STEREOSCOPY__H
#define STEREOSCOPY__H

#include "sfm_types.h"

struct xyz_coords_t *get_3d_object_coords(struct xyz_coords_t *coords_1, double angle_1,
                                          struct xyz_coords_t *coords_2, double angle_2);

#endif //STEREOSCOPY__H
