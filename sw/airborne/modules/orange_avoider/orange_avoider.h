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

#include <stdint.h>

/* ---------------- Green-follow settings ---------------- */
extern float oa_green_roi_frac_threshold;     // 0..1, ROI1 bottom
extern float oa_tree_roi2_frac_threshold;     // 0..1, ROI2 top (optional)
extern float oa_search_yaw_increment_deg;     // deg per periodic call
extern float oa_forward_step_m;              // meters per step

/* ---------------- Opticflow obstacle settings ---------------- */
extern float oa_of_noise_max;                // quality field is noise_measurement (lower is better)
extern float oa_of_div_thresh;               // divergence threshold
extern float oa_of_flow_mag_thresh;          // flow magnitude threshold (px-ish units from module)
extern float oa_of_close_min_speed_mps;      // only trust divergence when translating
extern uint8_t oa_of_close_n;                // debounce frames

extern float oa_of_avoid_time_s;             // seconds to run OF_AVOID before returning to greenfollow
extern float oa_of_retreat_m;                // retreat distance on OF trigger
extern float oa_of_turn_step_deg;            // yaw step (deg) while avoiding
extern float oa_of_goal_step_m;              // forward step while avoiding (GOAL)
extern float oa_of_traj_step_m;              // forward step while avoiding (TRAJECTORY)

/* module functions */
extern void orange_avoider_init(void);
extern void orange_avoider_periodic(void);

#endif
