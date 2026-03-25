/*
 * Copyright (C) Roland Meertens
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/orange_avoider/orange_avoider.c"
 * @author Roland Meertens
 * Example on how to use the colours detected to avoid orange pole in the cyberzoo
 * This module is an example module for the course AE4317 Autonomous Flight of Micro Air Vehicles at the TU Delft.
 * This module is used in combination with a color filter (cv_detect_color_object) and the navigation mode of the autopilot.
 * The avoidance strategy is to simply count the total number of orange pixels. When above a certain percentage threshold,
 * (given by color_count_frac) we assume that there is an obstacle and we turn.
 *
 * The color filter settings are set using the cv_detect_color_object. This module can run multiple filters simultaneously
 * so you have to define which filter to use with the ORANGE_AVOIDER_VISUAL_DETECTION_ID setting.
 */

 
#include "modules/orange_avoider/orange_avoider.h"

#include "firmwares/rotorcraft/navigation.h"
#include "generated/airframe.h"
#include "generated/flight_plan.h"

#include "modules/core/abi.h"
#include "state.h"
#include "mcu_periph/sys_time.h"

#include <stdio.h>
#include <stdbool.h>
#include <math.h>

/* ---------------- Logging ---------------- */
#ifndef ORANGE_AVOIDER_VERBOSE
#define ORANGE_AVOIDER_VERBOSE 1
#endif
#define FUNCTION __func__

#define FUNCTION __FUNCTION__
#define PRINT(fmt, ...) fprintf(stderr, "[orange_avoider->%s()] " fmt, FUNCTION, ##__VA_ARGS__)
#if ORANGE_AVOIDER_VERBOSE
#define VERBOSE_PRINT(...) PRINT(__VA_ARGS__)
#else
#define VERBOSE_PRINT(...)
#endif

/* ---------------- Defaults (can be overridden by airframe defines) ---------------- */
#ifndef OA_GREEN_ROI_FRAC_THRESHOLD
#define OA_GREEN_ROI_FRAC_THRESHOLD 0.70f
#endif

#ifndef OA_TREE_ROI2_FRAC_THRESHOLD
#define OA_TREE_ROI2_FRAC_THRESHOLD 0.35f
#endif

#ifndef OA_SEARCH_YAW_INCREMENT_DEG
#define OA_SEARCH_YAW_INCREMENT_DEG 10.f
#endif

#ifndef OA_FORWARD_STEP_M
#define OA_FORWARD_STEP_M 0.20f
#endif

#ifndef OA_OF_NOISE_MAX
#define OA_OF_NOISE_MAX 0.7f
#endif

#ifndef OA_OF_DIV_THRESH
#define OA_OF_DIV_THRESH 0.06f
#endif

#ifndef OA_OF_FLOW_MAG_THRESH
#define OA_OF_FLOW_MAG_THRESH 500.f
#endif

#ifndef OA_OF_CLOSE_MIN_SPEED_MPS
#define OA_OF_CLOSE_MIN_SPEED_MPS 0.05f
#endif

#ifndef OA_OF_CLOSE_N
#define OA_OF_CLOSE_N 2
#endif

#ifndef OA_OF_AVOID_TIME_S
#define OA_OF_AVOID_TIME_S 1.2f
#endif

#ifndef OA_OF_RETREAT_M
#define OA_OF_RETREAT_M 0.35f
#endif

#ifndef OA_OF_TURN_STEP_DEG
#define OA_OF_TURN_STEP_DEG 6.f
#endif

#ifndef OA_OF_GOAL_STEP_M
#define OA_OF_GOAL_STEP_M 0.35f
#endif

#ifndef OA_OF_TRAJ_STEP_M
#define OA_OF_TRAJ_STEP_M 0.70f
#endif

/* ---------------- Exposed settings variables ---------------- */
float oa_green_roi_frac_threshold  = OA_GREEN_ROI_FRAC_THRESHOLD;
float oa_tree_roi2_frac_threshold  = OA_TREE_ROI2_FRAC_THRESHOLD;
float oa_search_yaw_increment_deg  = OA_SEARCH_YAW_INCREMENT_DEG;
float oa_forward_step_m            = OA_FORWARD_STEP_M;

float oa_of_noise_max              = OA_OF_NOISE_MAX;
float oa_of_div_thresh             = OA_OF_DIV_THRESH;
float oa_of_flow_mag_thresh        = OA_OF_FLOW_MAG_THRESH;
float oa_of_close_min_speed_mps    = OA_OF_CLOSE_MIN_SPEED_MPS;
uint8_t oa_of_close_n              = OA_OF_CLOSE_N;

float oa_of_avoid_time_s           = OA_OF_AVOID_TIME_S;
float oa_of_retreat_m              = OA_OF_RETREAT_M;
float oa_of_turn_step_deg          = OA_OF_TURN_STEP_DEG;
float oa_of_goal_step_m            = OA_OF_GOAL_STEP_M;
float oa_of_traj_step_m            = OA_OF_TRAJ_STEP_M;

/* ---------------- Internal navigation state (green follow) ---------------- */
enum gf_state_t {
  GF_SAFE = 0,
  GF_SEARCH = 1
};
static enum gf_state_t gf_state = GF_SEARCH;

/* ---------------- Mode supervisor ---------------- */
enum oa_mode_t {
  OA_MODE_GREENFOLLOW = 0,
  OA_MODE_OF_AVOID = 1
};
static enum oa_mode_t oa_mode = OA_MODE_GREENFOLLOW;
static uint32_t of_avoid_until_us = 0;
static bool of_avoid_just_entered = false;

/* ---------------- Shared perception (from ABI) ---------------- */
#ifndef ORANGE_AVOIDER_VISUAL_DETECTION_ID
#define ORANGE_AVOIDER_VISUAL_DETECTION_ID ABI_BROADCAST
#endif

static abi_event color_detection_ev;
static abi_event opticflow_ev;

/* VISUAL_DETECTION is (pixel_x, pixel_y, pixel_width, pixel_height, quality, extra)
 * Your cv_detect_color_object sends:
 *   pixel_width  = roi_count
 *   pixel_height = roi_area
 *   quality      = roi2_count
 *   extra        = roi2_area
 */
static int32_t roi_count = 0;
static int32_t roi_area  = 0;
static int32_t roi2_count = 0;
static int32_t roi2_area  = 0;

static void color_detection_cb(uint8_t sender_id,
                               int16_t pixel_x, int16_t pixel_y,
                               int16_t pixel_width, int16_t pixel_height,
                               int32_t quality, int16_t extra)
{
  (void)sender_id; (void)pixel_x; (void)pixel_y;

  roi_count  = (int32_t)pixel_width;
  roi_area   = (int32_t)pixel_height;
  roi2_count = (int32_t)quality;
  roi2_area  = (int32_t)extra;
}

/* ---------------- Opticflow last values ---------------- */
static uint32_t of_msg_cnt = 0;
static float of_noise = 1.f;          // quality field = noise_measurement (lower is better)
static float of_div_size = 0.f;       // divergence field
static float of_div_filt = 0.f;
static int of_flow_x_last = 0;
static int of_flow_y_last = 0;
static uint8_t of_close_cnt = 0;

static void opticflow_cb(uint8_t sender_id,
                         uint32_t stamp,
                         int flow_x, int flow_y,
                         int flow_der_x, int flow_der_y,
                         float quality,
                         float divergence)
{
  (void)sender_id; (void)stamp;
  (void)flow_der_x; (void)flow_der_y;

  of_msg_cnt++;
  of_flow_x_last = flow_x;
  of_flow_y_last = flow_y;
  of_noise = quality;
  of_div_size = divergence;
}

/* ---------------- Helpers (waypoint + heading) ---------------- */
static uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters)
{
  const float heading = stateGetNedToBodyEulers_f()->psi;
  new_coor->x = stateGetPositionEnu_i()->x + POS_BFP_OF_REAL(sinf(heading) * distanceMeters);
  new_coor->y = stateGetPositionEnu_i()->y + POS_BFP_OF_REAL(cosf(heading) * distanceMeters);
  return false;
}

static uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor)
{
  waypoint_set_xy_i(waypoint, new_coor->x, new_coor->y);
  return false;
}

static uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters)
{
  struct EnuCoor_i new_coor;
  calculateForwards(&new_coor, distanceMeters);
  moveWaypoint(waypoint, &new_coor);
  return false;
}

static uint8_t increase_nav_heading(float incrementDegrees)
{
  float new_heading = stateGetNedToBodyEulers_f()->psi + RadOfDeg(incrementDegrees);
  FLOAT_ANGLE_NORMALIZE(new_heading);
  nav.heading = new_heading;
  return false;
}

/* ---------------- Module init/periodic ---------------- */
void orange_avoider_init(void)
{
  AbiBindMsgVISUAL_DETECTION(ORANGE_AVOIDER_VISUAL_DETECTION_ID, &color_detection_ev, color_detection_cb);
  AbiBindMsgOPTICAL_FLOW(ABI_BROADCAST, &opticflow_ev, opticflow_cb);

  gf_state = GF_SEARCH;
  oa_mode = OA_MODE_GREENFOLLOW;
  of_avoid_until_us = 0;
  of_avoid_just_entered = false;

  VERBOSE_PRINT("Init. GreenThr=%f TreeThr=%f\n", oa_green_roi_frac_threshold, oa_tree_roi2_frac_threshold);
}

void orange_avoider_periodic(void)
{
  if (!autopilot_in_flight()) {
    return;
  }

  /* --- Compute green fractions --- */
  float green_frac = 0.f;
  float top_frac = 0.f;

  if (roi_area > 0)  { green_frac = (float)roi_count  / (float)roi_area; }
  if (roi2_area > 0) { top_frac   = (float)roi2_count / (float)roi2_area; }

  const bool green_ok = (green_frac >= oa_green_roi_frac_threshold);
  const bool tree_like = (top_frac >= oa_tree_roi2_frac_threshold);

  /* --- Opticflow proximity (for ALL obstacles) --- */
  const bool have_of = (of_msg_cnt > 5);
  const bool of_good = have_of && (of_noise < oa_of_noise_max);

  struct EnuCoor_f *vel = stateGetSpeedEnu_f();
  const float vxy = sqrtf(vel->x * vel->x + vel->y * vel->y);
  const bool translating = (vxy > oa_of_close_min_speed_mps);

  const float flow_mag = sqrtf((float)of_flow_x_last * (float)of_flow_x_last +
                              (float)of_flow_y_last * (float)of_flow_y_last);

  if (of_good && translating) {
    of_div_filt = 0.7f * of_div_filt + 0.3f * of_div_size;
  } else {
    of_div_filt *= 0.9f;
  }

  const bool close_soft = translating &&
    (fabsf(of_div_filt) > oa_of_div_thresh || flow_mag > oa_of_flow_mag_thresh);

  struct FloatRates *rates = stateGetBodyRates_f();
  const bool not_turning_fast = fabsf(rates->r) < 0.4f;
  const bool close_hard = of_good && not_turning_fast && (flow_mag > (oa_of_flow_mag_thresh * 2.2f));

  const bool close_now = of_good && (close_soft || close_hard);

  /* debounce with decay */
  if (close_now) {
    if (of_close_cnt < oa_of_close_n) { of_close_cnt++; }
  } else {
    if (of_close_cnt > 0) { of_close_cnt--; }
  }

  const bool obstacle_of = (of_close_cnt >= oa_of_close_n);


  /* --- Mode switching: enter timed OF_AVOID --- */
  const uint32_t now_us = get_sys_time_usec();

  if (oa_mode != OA_MODE_OF_AVOID && obstacle_of) {
    oa_mode = OA_MODE_OF_AVOID;
    of_avoid_until_us = now_us + (uint32_t)(oa_of_avoid_time_s * 1e6f);
    of_avoid_just_entered = true;

    VERBOSE_PRINT("ENTER OF_AVOID: flow_mag=%f div_filt=%f noise=%f (until +%fs)\n",
                  flow_mag, of_div_filt, of_noise, oa_of_avoid_time_s);
  }

  if (oa_mode == OA_MODE_OF_AVOID) {

    if (of_avoid_just_entered) {
      /* retreat once to create spacing */
      waypoint_move_here_2d(WP_GOAL);
      waypoint_move_here_2d(WP_TRAJECTORY);
      moveWaypointForward(WP_GOAL, -oa_of_retreat_m);
      moveWaypointForward(WP_TRAJECTORY, -oa_of_retreat_m);
      of_avoid_just_entered = false;
    }

    /* turn away from dominant flow side */
    const float turn = (of_flow_x_last > 0) ? -oa_of_turn_step_deg : oa_of_turn_step_deg;
    increase_nav_heading(turn);

    /* keep moving (small) while avoiding */
    moveWaypointForward(WP_TRAJECTORY, oa_of_traj_step_m);
    moveWaypointForward(WP_GOAL, oa_of_goal_step_m);

    if ((int32_t)(now_us - of_avoid_until_us) >= 0) {
      oa_mode = OA_MODE_GREENFOLLOW;
      gf_state = GF_SEARCH; /* re-acquire green after avoidance */
      VERBOSE_PRINT("EXIT OF_AVOID -> GREENFOLLOW\n");
    }

    return; /* do not run greenfollow in avoid mode */
  }

  /* ---------------- GREENFOLLOW mode (quick green evaluation) ---------------- */
  VERBOSE_PRINT("GF mode: green=%0.2f(top=%0.2f) green_ok=%d tree_like=%d of(close=%d cnt=%u mag=%0.1f div=%0.3f)\n",
                green_frac, top_frac, (int)green_ok, (int)tree_like,
                (int)obstacle_of, (unsigned)of_close_cnt, flow_mag, of_div_filt);
  
  VERBOSE_PRINT("OF: cnt=%lu good=%d noise=%0.2f div=%0.3f divf=%0.3f flow=(%d,%d) mag=%0.1f close_cnt=%u\n",
  (unsigned long)of_msg_cnt, (int)of_good, of_noise, of_div_size, of_div_filt,
  of_flow_x_last, of_flow_y_last, flow_mag, (unsigned)of_close_cnt);

  switch (gf_state) {

    case GF_SAFE:
      if (tree_like) {
        waypoint_move_here_2d(WP_GOAL);
        waypoint_move_here_2d(WP_TRAJECTORY);
        gf_state = GF_SEARCH;
      } else if (green_ok) {
        moveWaypointForward(WP_TRAJECTORY, oa_forward_step_m);
        moveWaypointForward(WP_GOAL, oa_forward_step_m);
      } else {
        waypoint_move_here_2d(WP_GOAL);
        waypoint_move_here_2d(WP_TRAJECTORY);
        gf_state = GF_SEARCH;
      }
      break;

    case GF_SEARCH:
    default:
      waypoint_move_here_2d(WP_GOAL);
      waypoint_move_here_2d(WP_TRAJECTORY);

      if (!tree_like && green_ok) {
        gf_state = GF_SAFE;
      } else {
        increase_nav_heading(oa_search_yaw_increment_deg);
      }
      break;
  }
}
