#ifndef CV_DETECT_DUAL_H
#define CV_DETECT_DUAL_H

#include "std.h"
#include "modules/computer_vision/lib/vision/image.h"

/**
 * cv_detect_dual.h
 *
 * Dual-camera color detector for Paparazzi UAV on Bebop drone.
 *
 * - FRONT camera: detects ORANGE obstacles
 * - BOTTOM camera: detects GREEN mat (landing zone / safe area)
 *
 * Uses YCbCr color space (native to Bebop camera output).
 * Results are published via ABI messaging to other modules
 * (e.g. orange_avoider, landing detector).
 */

/* ---------------------------------------------------------------
 * ORANGE DETECTOR (front camera) — configurable via airframe XML
 * --------------------------------------------------------------- */
#ifndef ORANGE_DETECTOR_LUM_MIN
#define ORANGE_DETECTOR_LUM_MIN   30
#endif
#ifndef ORANGE_DETECTOR_LUM_MAX
#define ORANGE_DETECTOR_LUM_MAX   190
#endif
#ifndef ORANGE_DETECTOR_CB_MIN
#define ORANGE_DETECTOR_CB_MIN    70
#endif
#ifndef ORANGE_DETECTOR_CB_MAX
#define ORANGE_DETECTOR_CB_MAX    130
#endif
#ifndef ORANGE_DETECTOR_CR_MIN
#define ORANGE_DETECTOR_CR_MIN    150
#endif
#ifndef ORANGE_DETECTOR_CR_MAX
#define ORANGE_DETECTOR_CR_MAX    190
#endif

/* ---------------------------------------------------------------
 * GREEN DETECTOR (bottom camera) — configurable via airframe XML
 * Green in YCbCr: low Cr, moderate-high Cb, moderate luminance
 * --------------------------------------------------------------- */
#ifndef GREEN_DETECTOR_LUM_MIN
#define GREEN_DETECTOR_LUM_MIN    40
#endif
#ifndef GREEN_DETECTOR_LUM_MAX
#define GREEN_DETECTOR_LUM_MAX    200
#endif
#ifndef GREEN_DETECTOR_CB_MIN
#define GREEN_DETECTOR_CB_MIN     80
#endif
#ifndef GREEN_DETECTOR_CB_MAX
#define GREEN_DETECTOR_CB_MAX     140
#endif
#ifndef GREEN_DETECTOR_CR_MIN
#define GREEN_DETECTOR_CR_MIN     80
#endif
#ifndef GREEN_DETECTOR_CR_MAX
#define GREEN_DETECTOR_CR_MAX     120
#endif

/* Minimum pixel count to consider a detection valid */
#ifndef ORANGE_DETECTOR_MIN_PIXELS
#define ORANGE_DETECTOR_MIN_PIXELS  100
#endif
#ifndef GREEN_DETECTOR_MIN_PIXELS
#define GREEN_DETECTOR_MIN_PIXELS   200
#endif

/* Draw bounding boxes / colored overlays on video stream (debug) */
#ifndef ORANGE_DETECTOR_DRAW
#define ORANGE_DETECTOR_DRAW  TRUE
#endif
#ifndef GREEN_DETECTOR_DRAW
#define GREEN_DETECTOR_DRAW   TRUE
#endif

/* ---------------------------------------------------------------
 * Result structs
 * --------------------------------------------------------------- */

/** Detection result for one color target */
struct ColorDetectionResult {
  bool  detected;        ///< TRUE if enough pixels found
  int   cx;             ///< Centroid X (pixels from left)
  int   cy;             ///< Centroid Y (pixels from top)
  int   pixel_count;    ///< Number of matching pixels
  float frac_of_frame;  ///< Fraction of total frame area (0.0 - 1.0)

  /* For orange: split left/right to decide turn direction */
  int   pixels_left;    ///< Matching pixels in left half of frame
  int   pixels_right;   ///< Matching pixels in right half of frame
};

extern struct ColorDetectionResult orange_detection; ///< Latest orange result
extern struct ColorDetectionResult green_detection;  ///< Latest green result

/* ---------------------------------------------------------------
 * Paparazzi module interface
 * --------------------------------------------------------------- */
extern void cv_detect_dual_init(void);
extern void cv_detect_dual_periodic(void);  ///< Called every control loop

/* Internal camera callbacks (registered in _init) */
extern struct image_t *cv_detect_orange_cb(struct image_t *img, uint8_t camera_id);
extern struct image_t *cv_detect_green_cb(struct image_t *img, uint8_t camera_id);

#endif /* CV_DETECT_DUAL_H */
