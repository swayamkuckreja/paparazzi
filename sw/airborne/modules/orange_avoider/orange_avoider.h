/*
 * Copyright (C) Roland Meertens
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/orange_avoider/orange_avoider.h"
 * @author Roland Meertens
 * Example on how to use the colours detected to avoid orange pole in the cyberzoo
 */
#ifndef ORANGE_AVOIDER_H
#define ORANGE_AVOIDER_H

#ifdef __cplusplus
extern "C" {
#endif

// settings (tunable via GCS if listed in the module xml)
extern float oa_green_roi_frac_threshold;
extern float oa_search_yaw_increment_deg;
extern float oa_forward_step_m;

// orange detection fraction threshold (used by merged orange_avoider.c)
extern float oa_color_count_frac;

// module hooks
extern void orange_avoider_init(void);
extern void orange_avoider_periodic(void);

#ifdef __cplusplus
}
#endif

#endif // ORANGE_AVOIDER_H