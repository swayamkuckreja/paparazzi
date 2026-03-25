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
#include "state.h"
#include "modules/core/abi.h"
#include <time.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include "modules/computer_vision/cv.h"
#include "modules/computer_vision/lib/vision/image.h"
#include <pthread.h>



#include "generated/flight_plan.h"

#define ORANGE_AVOIDER_VERBOSE TRUE

#define PRINT(string,...) fprintf(stderr, "[orange_avoider->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#if ORANGE_AVOIDER_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

static uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters);
static uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters);
static uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor);
static uint8_t increase_nav_heading(float incrementDegrees);
static uint8_t chooseRandomIncrementAvoidance(void);
static abi_event opticflow_ev;
static uint32_t of_msg_cnt = 0;

static pthread_mutex_t oa_vis_mutex;
static float oa_prox01 = 0.f;  // 0..1 proximity value for visualization
static struct video_listener *oa_vis_listener = NULL;

static struct image_t *orange_avoider_vis_cb(struct image_t *img, uint8_t cam_id);


static void opticflow_cb(uint8_t sender_id,
                         uint32_t stamp,
                         int flow_x, int flow_y,
                         int flow_der_x, int flow_der_y,
                         float quality,
                         float divergence);




// Opticflow last values
static float of_div_size = 0.f;     // last received div_size
static float of_noise    = 1.f;     // noise_measurement (lower is better), start "bad"
static int   of_flow_x_last = 0;    // for directional turn
static int of_flow_y_last = 0;

// Proximity detection tuning
static float of_div_filt = 0.f;
static uint8_t of_close_cnt = 0;

static float of_noise_max   = 0.7f;   // stricter than 0.8
static float of_div_thresh  = 0.06f;  // lower => detects earlier
static float of_flow_mag_thresh = 600.f;   // tune (start ~250-400 in your sim)
// static float of_yawrate_max = 0.7f;   // rad/s (~40 deg/s)

#define OF_CLOSE_N 2                  // need N consecutive "close" frames


static void opticflow_cb(uint8_t sender_id,
                         uint32_t stamp,
                         int flow_x,
                         int flow_y,
                         int flow_der_x,
                         int flow_der_y,
                         float quality,
                         float divergence)
{
  (void)sender_id; (void)stamp;
  (void)flow_y;
  (void)flow_der_x; (void)flow_der_y;

  of_msg_cnt++;
  of_flow_x_last = flow_x;
  of_flow_y_last = flow_y;

  // opticflow_module sends noise_measurement in the "quality" field
  of_noise = quality;

  // opticflow_module sends div_size as the last field
  of_div_size = divergence;
}

enum navigation_state_t {
  SAFE,
  OBSTACLE_FOUND,
  SEARCH_FOR_SAFE_HEADING,
  OUT_OF_BOUNDS
};

// define settings
float oa_color_count_frac = 0.18f;

// define and initialise global variables
enum navigation_state_t navigation_state = SEARCH_FOR_SAFE_HEADING;
int32_t color_count = 0;                // orange color count from color filter for obstacle detection
int16_t obstacle_free_confidence = 0;   // a measure of how certain we are that the way ahead is safe.
float heading_increment = 5.f;          // heading angle increment [deg]
float maxDistance = 2.25;               // max waypoint displacement [m]

const int16_t max_trajectory_confidence = 5; // number of consecutive negative object detections to be sure we are obstacle free

/*
 * This next section defines an ABI messaging event (http://wiki.paparazziuav.org/wiki/ABI), necessary
 * any time data calculated in another module needs to be accessed. Including the file where this external
 * data is defined is not enough, since modules are executed parallel to each other, at different frequencies,
 * in different threads. The ABI event is triggered every time new data is sent out, and as such the function
 * defined in this file does not need to be explicitly called, only bound in the init function
 */
#ifndef ORANGE_AVOIDER_VISUAL_DETECTION_ID
#define ORANGE_AVOIDER_VISUAL_DETECTION_ID ABI_BROADCAST
#endif
static abi_event color_detection_ev;
static void color_detection_cb(uint8_t __attribute__((unused)) sender_id,
                               int16_t __attribute__((unused)) pixel_x, int16_t __attribute__((unused)) pixel_y,
                               int16_t __attribute__((unused)) pixel_width, int16_t __attribute__((unused)) pixel_height,
                               int32_t quality, int16_t __attribute__((unused)) extra)
{
  color_count = quality;
}

/*
 * Initialisation function, setting the colour filter, random seed and heading_increment
 */
void orange_avoider_init(void)
{
  srand(time(NULL));
  chooseRandomIncrementAvoidance();

  AbiBindMsgVISUAL_DETECTION(ORANGE_AVOIDER_VISUAL_DETECTION_ID, &color_detection_ev, color_detection_cb);
  AbiBindMsgOPTICAL_FLOW(ABI_BROADCAST, &opticflow_ev, opticflow_cb);

  pthread_mutex_init(&oa_vis_mutex, NULL);
  oa_vis_listener = cv_add_to_device(&front_camera, orange_avoider_vis_cb, 10, 0);

}


/*
 * Function that checks it is safe to move forwards, and then moves a waypoint forward or changes the heading
 */
void orange_avoider_periodic(void)
{
  if (!autopilot_in_flight()) {
    return;
  }

  struct EnuCoor_f *vel = stateGetSpeedEnu_f();
  float vxy = sqrtf(vel->x*vel->x + vel->y*vel->y);
  bool translating = vxy > 0.05f;


  // ---- Color obstacle detection ----
  int32_t color_count_threshold =
      (int32_t)(oa_color_count_frac * front_camera.output_size.w * front_camera.output_size.h);

  bool obstacle_detected_color = (color_count >= color_count_threshold);

  


    // ---- Opticflow obstacle detection (nearby obstacle proxy) ----
  bool have_of = (of_msg_cnt > 5);
  bool of_good = have_of && (of_noise < of_noise_max);

  float yaw_rate = stateGetBodyRates_f()->r;            // rad/s
  bool not_turning_fast = fabsf(yaw_rate) < 0.4f;       // rad/s (tune)

  // flow magnitude proxy
  float flow_mag = sqrtf((float)of_flow_x_last * (float)of_flow_x_last +
                        (float)of_flow_y_last * (float)of_flow_y_last);

  // update divergence filter (only when moving)
  if (of_good && translating) {
    of_div_filt = 0.7f * of_div_filt + 0.3f * of_div_size;
  } else {
    of_div_filt *= 0.9f;
  }

  // soft close: when translating
  bool close_soft = translating &&
    (fabsf(of_div_filt) > of_div_thresh || flow_mag > of_flow_mag_thresh);

  // hard close: even if nearly stopped, but only if not turning fast
  bool close_hard = not_turning_fast && (flow_mag > 1200.f); // tune

  bool close_now = of_good && (close_soft || close_hard);

  // debounce
  if (close_now) {
    if (of_close_cnt < OF_CLOSE_N) { of_close_cnt++; }
  } else {
    of_close_cnt = 0;
  }

  bool obstacle_detected_flow = (of_close_cnt >= OF_CLOSE_N);





  bool obstacle_detected = obstacle_detected_color || obstacle_detected_flow;

  VERBOSE_PRINT("### NEW BUILD ### Color=%d thr=%d state=%d\n",
                color_count, color_count_threshold, navigation_state);
  
  VERBOSE_PRINT("OF cnt=%lu noise=%f div=%f div_filt=%f of_good=%d flow=(%d,%d) close_cnt=%d obs_of=%d\n",
    (unsigned long)of_msg_cnt, of_noise, of_div_size, of_div_filt, of_good,
    of_flow_x_last, of_flow_y_last, of_close_cnt, obstacle_detected_flow);

  VERBOSE_PRINT("vxy=%f translating=%d flow_mag=%f\n", vxy, translating, flow_mag);

  VERBOSE_PRINT("yaw_rate=%f rad/s\n", yaw_rate);



  // ---- Confidence update ----
  if (!obstacle_detected) {
    obstacle_free_confidence++;
  } else {
    obstacle_free_confidence -= 2;
  }

  Bound(obstacle_free_confidence, 0, max_trajectory_confidence);

  float moveDistance = fminf(maxDistance, 0.2f * obstacle_free_confidence);

  // Map flow_mag to 0..1 for visualization (tune these two numbers)
  const float FLOW_MIN = 100.f;   // "far"
  const float FLOW_MAX = 2000.f;  // "very close"
  float prox = (flow_mag - FLOW_MIN) / (FLOW_MAX - FLOW_MIN);
  if (prox < 0.f) prox = 0.f;
  if (prox > 1.f) prox = 1.f;

  pthread_mutex_lock(&oa_vis_mutex);
  oa_prox01 = of_good ? prox : 0.f;
  pthread_mutex_unlock(&oa_vis_mutex);


  // ---- State machine ----
  switch (navigation_state) {

    case SAFE:
      moveWaypointForward(WP_TRAJECTORY, 1.5f * moveDistance);

      if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
        navigation_state = OUT_OF_BOUNDS;

      } else if (obstacle_free_confidence == 0) {
        navigation_state = OBSTACLE_FOUND;

      } else {
        moveWaypointForward(WP_GOAL, moveDistance);
      }
      break;

    
    case OBSTACLE_FOUND: {
      waypoint_move_here_2d(WP_GOAL);
      waypoint_move_here_2d(WP_TRAJECTORY);
      // back up a bit to create distance to obstacle
      float retreat_m = 0.3f; 
      moveWaypointForward(WP_GOAL,     -retreat_m);
      moveWaypointForward(WP_TRAJECTORY, -retreat_m);

      // Direction choice:
      // - If opticflow triggered, turn based on flow_x sign (turn away from dominant flow side)
      // - Otherwise, fall back to random (color-only obstacle)
      if (obstacle_detected_flow) {
        heading_increment = (of_flow_x_last > 0) ? -5.f : 5.f;
        VERBOSE_PRINT("OF-based avoidance: flow_x=%d -> heading_increment=%f\n",
                      of_flow_x_last, heading_increment);
      } else {
        chooseRandomIncrementAvoidance();
      }

      navigation_state = SEARCH_FOR_SAFE_HEADING;
      break;
    }

    case SEARCH_FOR_SAFE_HEADING:
      increase_nav_heading(heading_increment);

      if (obstacle_free_confidence >= 2) {
        navigation_state = SAFE;
      }
      break;

    case OUT_OF_BOUNDS:
      increase_nav_heading(heading_increment);
      moveWaypointForward(WP_TRAJECTORY, 1.5f);

      if (obstacle_detected_flow) {
        navigation_state = OBSTACLE_FOUND;
        obstacle_free_confidence = 0;
        break;
      }

      if (InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
        increase_nav_heading(heading_increment);
        obstacle_free_confidence = 0;
        navigation_state = SEARCH_FOR_SAFE_HEADING;
      }
      break;

    default:
      break;
  }
}

static struct image_t *orange_avoider_vis_cb(struct image_t *img, uint8_t cam_id)
{
  (void)cam_id; // not used

  float prox;
  pthread_mutex_lock(&oa_vis_mutex);
  prox = oa_prox01;
  pthread_mutex_unlock(&oa_vis_mutex);

  const uint8_t U = 128, V = 128;
  uint8_t Y = (uint8_t)(30 + prox * 200);

  int bar_w = 12;
  if (bar_w > img->w) bar_w = img->w;

  uint8_t *buf = (uint8_t *)img->buf;
  for (int y = 0; y < img->h; y++) {
    for (int x = 0; x < bar_w; x++) {
      uint8_t *p = buf + y * img->w * 2 + (x / 2) * 4; // U Y0 V Y1
      p[0] = U; p[2] = V;
      if ((x & 1) == 0) p[1] = Y; else p[3] = Y;
    }
  }
  return img;
}




/*
 * Increases the NAV heading. Assumes heading is an INT32_ANGLE. It is bound in this function.
 */
uint8_t increase_nav_heading(float incrementDegrees)
{
  float new_heading = stateGetNedToBodyEulers_f()->psi + RadOfDeg(incrementDegrees);

  // normalize heading to [-pi, pi]
  FLOAT_ANGLE_NORMALIZE(new_heading);

  // set heading, declared in firmwares/rotorcraft/navigation.h
  nav.heading = new_heading;

  VERBOSE_PRINT("Increasing heading to %f\n", DegOfRad(new_heading));
  return false;
}

/*
 * Calculates coordinates of distance forward and sets waypoint 'waypoint' to those coordinates
 */
uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters)
{
  struct EnuCoor_i new_coor;
  calculateForwards(&new_coor, distanceMeters);
  moveWaypoint(waypoint, &new_coor);
  return false;
}

/*
 * Calculates coordinates of a distance of 'distanceMeters' forward w.r.t. current position and heading
 */
uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters)
{
  float heading  = stateGetNedToBodyEulers_f()->psi;

  // Now determine where to place the waypoint you want to go to
  new_coor->x = stateGetPositionEnu_i()->x + POS_BFP_OF_REAL(sinf(heading) * (distanceMeters));
  new_coor->y = stateGetPositionEnu_i()->y + POS_BFP_OF_REAL(cosf(heading) * (distanceMeters));
  VERBOSE_PRINT("Calculated %f m forward position. x: %f  y: %f based on pos(%f, %f) and heading(%f)\n", distanceMeters,	
                POS_FLOAT_OF_BFP(new_coor->x), POS_FLOAT_OF_BFP(new_coor->y),
                stateGetPositionEnu_f()->x, stateGetPositionEnu_f()->y, DegOfRad(heading));
  return false;
}

/*
 * Sets waypoint 'waypoint' to the coordinates of 'new_coor'
 */
uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor)
{
  VERBOSE_PRINT("Moving waypoint %d to x:%f y:%f\n", waypoint, POS_FLOAT_OF_BFP(new_coor->x),
                POS_FLOAT_OF_BFP(new_coor->y));
  waypoint_move_xy_i(waypoint, new_coor->x, new_coor->y);
  return false;
}

/*
 * Sets the variable 'heading_increment' randomly positive/negative
 */
uint8_t chooseRandomIncrementAvoidance(void)
{
  // Randomly choose CW or CCW avoiding direction
  if (rand() % 2 == 0) {
    heading_increment = 5.f;
    VERBOSE_PRINT("Set avoidance increment to: %f\n", heading_increment);
  } else {
    heading_increment = -5.f;
    VERBOSE_PRINT("Set avoidance increment to: %f\n", heading_increment);
  }
  return false;
}

