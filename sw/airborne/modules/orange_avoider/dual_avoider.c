/**
 * dual_avoider.c
 *
 * Navigation/avoidance module that uses:
 *   - ORANGE detection (front camera) → steer away from obstacles
 *   - GREEN detection  (bottom camera) → recognize safe landing mat
 *
 * Behaviour logic:
 * ----------------
 * 1. If orange is detected ahead:
 *      - If more orange on RIGHT → turn LEFT
 *      - If more orange on LEFT  → turn RIGHT
 *      - If orange fills >40% of frame → stop and turn 180°
 *
 * 2. If green mat detected below (>30% of bottom frame):
 *      - Set a flag: safe_to_land = TRUE
 *      - Optionally trigger a landing sequence
 *
 * 3. If nothing detected → continue forward at cruise speed.
 *
 * This module subscribes to ABI messages published by cv_detect_dual.c
 */

#include "modules/computer_vision/cv_detect_dual.h"
#include "modules/core/abi.h"
#include "firmwares/rotorcraft/navigation.h"
#include "generated/flight_plan.h"
#include "autopilot.h"
#include <stdio.h>
#include <math.h>

/* ---------------------------------------------------------------
 * Tuning parameters (can be overridden in airframe XML)
 * --------------------------------------------------------------- */

/** Fraction of frame that triggers an emergency turn (0.0-1.0) */
#ifndef AVOIDER_ORANGE_CRITICAL_FRAC
#define AVOIDER_ORANGE_CRITICAL_FRAC   0.40f
#endif

/** Fraction of frame that triggers a normal steer (0.0-1.0) */
#ifndef AVOIDER_ORANGE_WARN_FRAC
#define AVOIDER_ORANGE_WARN_FRAC       0.10f
#endif

/** Fraction of bottom frame green needed to flag safe landing */
#ifndef AVOIDER_GREEN_LAND_FRAC
#define AVOIDER_GREEN_LAND_FRAC        0.30f
#endif

/** Turn increment per cycle in degrees */
#ifndef AVOIDER_TURN_STEP_DEG
#define AVOIDER_TURN_STEP_DEG          10.0f
#endif

/** Forward speed when no obstacle (m/s) */
#ifndef AVOIDER_CRUISE_SPEED
#define AVOIDER_CRUISE_SPEED           0.4f
#endif

/* ---------------------------------------------------------------
 * Internal state
 * --------------------------------------------------------------- */
typedef enum {
  AVOIDER_FORWARD,      /* Normal forward flight */
  AVOIDER_TURN_LEFT,    /* Turning left to avoid orange */
  AVOIDER_TURN_RIGHT,   /* Turning right to avoid orange */
  AVOIDER_EMERGENCY,    /* Large obstacle, turning 180 */
  AVOIDER_SAFE_LAND     /* Green mat detected below */
} AvoiderState;

static AvoiderState avoider_state = AVOIDER_FORWARD;
static float        heading_sp    = 0.0f;   /* Current heading setpoint (rad) */
static bool         safe_to_land  = false;  /* Green mat confirmed below */

/* ABI subscriber handles */
static abi_event orange_ev;
static abi_event green_ev;

/* ---------------------------------------------------------------
 * ABI callback: receives orange detection from front camera
 * --------------------------------------------------------------- */
static void orange_detection_cb(
    uint8_t sender_id __attribute__((unused)),
    int16_t pixel_x   __attribute__((unused)),
    int16_t pixel_y   __attribute__((unused)),
    int16_t pixel_width __attribute__((unused)),
    int16_t pixel_height __attribute__((unused)),
    int32_t quality,
    int16_t camera_id __attribute__((unused)))
{
  /* quality = pixel_count published by cv_detect_dual */
  /* We use the global struct for more detailed info    */

  if (!orange_detection.detected) {
    avoider_state = AVOIDER_FORWARD;
    return;
  }

  float frac = orange_detection.frac_of_frame;

  if (frac > AVOIDER_ORANGE_CRITICAL_FRAC) {
    /* Obstacle fills most of view — emergency U-turn */
    avoider_state = AVOIDER_EMERGENCY;
    printf("[AVOIDER] EMERGENCY: orange frac=%.2f, turning 180\n", frac);

  } else if (frac > AVOIDER_ORANGE_WARN_FRAC) {
    /* Decide turn direction based on which side has more orange */
    if (orange_detection.pixels_right > orange_detection.pixels_left) {
      avoider_state = AVOIDER_TURN_LEFT;
      printf("[AVOIDER] Orange on RIGHT, turning LEFT\n");
    } else {
      avoider_state = AVOIDER_TURN_RIGHT;
      printf("[AVOIDER] Orange on LEFT, turning RIGHT\n");
    }
  } else {
    /* Small amount of orange — keep going */
    avoider_state = AVOIDER_FORWARD;
  }
}

/* ---------------------------------------------------------------
 * ABI callback: receives green detection from bottom camera
 * --------------------------------------------------------------- */
static void green_detection_cb(
    uint8_t sender_id __attribute__((unused)),
    int16_t pixel_x   __attribute__((unused)),
    int16_t pixel_y   __attribute__((unused)),
    int16_t pixel_width __attribute__((unused)),
    int16_t pixel_height __attribute__((unused)),
    int32_t quality,
    int16_t camera_id __attribute__((unused)))
{
  if (!green_detection.detected) {
    safe_to_land = false;
    return;
  }

  if (green_detection.frac_of_frame > AVOIDER_GREEN_LAND_FRAC) {
    safe_to_land  = true;
    avoider_state = AVOIDER_SAFE_LAND;
    printf("[AVOIDER] GREEN mat detected below! frac=%.2f — safe to land.\n",
           green_detection.frac_of_frame);
  }
}

/* ---------------------------------------------------------------
 * Module init
 * --------------------------------------------------------------- */
void dual_avoider_init(void)
{
  avoider_state = AVOIDER_FORWARD;
  safe_to_land  = false;
  heading_sp    = 0.0f;

  /* Subscribe to orange detections (channel 1 = front camera) */
  AbiBindMsgVISUAL_DETECTION(COLOR_OBJECT_DETECTION1_ID, &orange_ev, orange_detection_cb);

  /* Subscribe to green detections (channel 2 = bottom camera) */
  AbiBindMsgVISUAL_DETECTION(COLOR_OBJECT_DETECTION2_ID, &green_ev,  green_detection_cb);

  printf("[dual_avoider] Initialized.\n");
}

/* ---------------------------------------------------------------
 * Periodic function — runs every control cycle
 * Translates avoider state into navigation commands
 * --------------------------------------------------------------- */
void dual_avoider_periodic(void)
{
  switch (avoider_state) {

    case AVOIDER_FORWARD:
      /* Fly forward at cruise speed, maintain current heading */
      nav_set_heading_rad(heading_sp);
      /* Set velocity via guidance — forward at cruise speed */
      /* In Paparazzi this is done via waypoint or velocity setpoint */
      break;

    case AVOIDER_TURN_LEFT:
      /* Increment heading left (negative = CCW in Paparazzi) */
      heading_sp -= RadOfDeg(AVOIDER_TURN_STEP_DEG);
      nav_set_heading_rad(heading_sp);
      /* Slow down while turning */
      break;

    case AVOIDER_TURN_RIGHT:
      heading_sp += RadOfDeg(AVOIDER_TURN_STEP_DEG);
      nav_set_heading_rad(heading_sp);
      break;

    case AVOIDER_EMERGENCY:
      /* Turn 180 degrees */
      heading_sp += RadOfDeg(180.0f);
      nav_set_heading_rad(heading_sp);
      avoider_state = AVOIDER_FORWARD; /* Resume after turn applied */
      break;

    case AVOIDER_SAFE_LAND:
      /*
       * Green mat confirmed below.
       * Trigger auto-landing via autopilot mode switch.
       * Uncomment the line below to actually land:
       */
      /* autopilot_set_mode(AP_MODE_GUIDED); */
      printf("[AVOIDER] Holding position above green mat.\n");
      break;

    default:
      avoider_state = AVOIDER_FORWARD;
      break;
  }
}

/* ---------------------------------------------------------------
 * Public getter: is it safe to land?
 * Can be called from flight plan or other modules.
 * --------------------------------------------------------------- */
bool dual_avoider_safe_to_land(void)
{
  return safe_to_land;
}
