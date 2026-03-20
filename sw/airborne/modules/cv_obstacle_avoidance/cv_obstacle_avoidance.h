#ifndef CV_OBSTACLE_AVOIDANCE_H
#define CV_OBSTACLE_AVOIDANCE_H

/**
 * cv_obstacle_avoidance.h
 * ========================
 * Onboard Paparazzi module – Bebop 2
 *
 * OBSTACLE DETECTION STRATEGY:
 * ─────────────────────────────
 *   PRIMARY   → Optical flow divergence (Y channel, block matching)
 *               Detects ANY obstacle regardless of colour or shape.
 *               Positive divergence = pixels expanding = object approaching.
 *
 *   SECONDARY → Sobel edge density (Y channel)
 *               Catches textureless obstacles (plain walls, white boards)
 *               that produce little optical flow.
 *
 *   REMOVED   → Floor colour segmentation (was unreliable for unknown obstacles)
 *
 * GATE DETECTION STRATEGY:
 * ─────────────────────────
 *   PRIMARY   → YCbCr colour blob (gate has a known, consistent colour)
 *   SECONDARY → Offboard Python sends refined detections via IvyBus
 *
 * IMAGE FORMAT: UYVY (native Bebop output, no RGB conversion anywhere)
 *   [0] U = Cb   [1] Y0   [2] V = Cr   [3] Y1
 *
 * GREEN gate in YCbCr:
 *   Both Cb and Cr are BELOW 128 (neutral midpoint).
 *   Defaults: Y 80–220, Cb 60–112, Cr 60–112.
 */

#include <stdint.h>
#include "modules/computer_vision/cv.h"
#include "modules/computer_vision/lib/vision/image.h"

/* -----------------------------------------------------------------------
 * Tunable parameters – override in airframe XML with <define> tags
 * --------------------------------------------------------------------- */

/** Forward cruise speed in GUIDED mode (m/s) */
extern float     oa_max_speed;

/** Heading step applied each avoid cycle (radians, ~23 deg default) */
extern float     oa_heading_rate;

/** Number of 30 Hz cycles to pause before choosing a turn direction */
extern uint16_t  oa_avoid_dwell_cycles;

/**
 * PRIMARY obstacle threshold – optical flow radial divergence.
 * Raise  → less sensitive (fewer false stops).
 * Lower  → more sensitive (reacts earlier to approaching objects).
 * Typical range: 0.015 – 0.060.
 */
extern float     oa_divergence_threshold;

/**
 * SECONDARY obstacle threshold – Sobel edge density (edges per pixel).
 * Only triggers when divergence alone is insufficient (e.g. blank walls).
 * Typical range: 0.20 – 0.35.
 */
extern float     oa_edge_density_threshold;

/**
 * Divergence must stay below this for at least one cycle before the
 * drone resumes forward flight after avoiding.  Slightly lower than
 * oa_divergence_threshold to add hysteresis and prevent rapid toggling.
 */
extern float     oa_divergence_clear_threshold;

/* -----------------------------------------------------------------------
 * GREEN GATE YCbCr thresholds
 * Calibrate with the YUVCalibrator tool in gate_detector.py.
 * --------------------------------------------------------------------- */
extern uint8_t oa_gate_y_min,  oa_gate_y_max;
extern uint8_t oa_gate_cb_min, oa_gate_cb_max;
extern uint8_t oa_gate_cr_min, oa_gate_cr_max;

/* -----------------------------------------------------------------------
 * Navigation state machine
 * --------------------------------------------------------------------- */
typedef enum {
    NAV_SAFE          = 0,  /**< Path clear – cruise forward              */
    NAV_OBSTACLE_NEAR = 1,  /**< Obstacle detected – brake to a stop      */
    NAV_AVOIDING      = 2,  /**< Rotating to find a clear heading         */
    NAV_GATE_FOUND    = 3,  /**< Gate visible – aligning heading          */
    NAV_GATE_APPROACH = 4,  /**< Aligned – accelerating toward gate       */
    NAV_GATE_PASS     = 5,  /**< Committing through gate opening          */
    NAV_EMERGENCY     = 6,  /**< All headings blocked – hover in place    */
} nav_state_t;

/* -----------------------------------------------------------------------
 * Vision results
 * Written by the CV callback (vision thread).
 * Read by the periodic navigation function (nav thread).
 * --------------------------------------------------------------------- */
typedef struct {

    /* ── Optical flow (PRIMARY obstacle signal) ──────────────────────── */
    float    divergence;        /**< Radial divergence; + = looming        */
    float    flow_left;         /**< Mean divergence in left  column third */
    float    flow_center;       /**< Mean divergence in centre column third */
    float    flow_right;        /**< Mean divergence in right column third */

    /* ── Sobel edge density per column third (SECONDARY obstacle signal) */
    float    edge_density_left;
    float    edge_density_center;
    float    edge_density_right;

    /* ── Gate detection ───────────────────────────────────────────────── */
    uint8_t  gate_detected;     /**< 1 = valid gate detection this frame   */
    float    gate_x_norm;       /**< Gate centre x  [-1 left … +1 right]  */
    float    gate_y_norm;       /**< Gate centre y  [-1 top  … +1 bottom] */
    float    gate_size_norm;    /**< Gate bbox width / frame width         */
    float    gate_confidence;   /**< Detection confidence [0, 1]           */
    uint8_t centre_dark;
} vision_results_t;

extern volatile vision_results_t oa_vision;
extern volatile nav_state_t      oa_nav_state;

/* -----------------------------------------------------------------------
 * Module interface
 * --------------------------------------------------------------------- */
void cv_obstacle_avoidance_init(void);
void cv_obstacle_avoidance_periodic(void);

/** CV pipeline callback – registered with cv_add_to_device() */
struct image_t *cv_oa_vision_cb(struct image_t *img, uint8_t camera_id);

/** Called when the offboard Python script sends a gate detection over IvyBus */
void oa_set_gate_detection(float x_norm, float y_norm,
                            float size_norm, float confidence);

#endif /* CV_OBSTACLE_AVOIDANCE_H */
