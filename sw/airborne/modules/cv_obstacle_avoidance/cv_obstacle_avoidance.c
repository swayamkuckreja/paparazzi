/**
 * cv_obstacle_avoidance.c
 * ========================
 * Onboard Paparazzi module – Bebop 2
 *
 * ═══════════════════════════════════════════════════════════════════
 *  WHAT CHANGED FROM THE PREVIOUS VERSION AND WHY
 * ═══════════════════════════════════════════════════════════════════
 *
 *  REMOVED: segment_floor_yuv()
 *    The old code counted green floor pixels to decide if a path was
 *    clear.  This only works when the obstacle blocks the floor view,
 *    and completely fails for same-coloured, head-height, or
 *    transparent obstacles.  It has been removed from the obstacle
 *    pipeline entirely.
 *
 *  PRIMARY OBSTACLE DETECTOR: compute_flow_divergence_y()
 *    Optical flow divergence works on ANY obstacle regardless of
 *    colour, shape, or size.  When an object approaches the camera,
 *    its pixels expand outward — this radial "divergence" signal is
 *    computed per column third so we know which side is blocked.
 *
 *  SECONDARY OBSTACLE DETECTOR: compute_edge_density_y()  (UNCHANGED)
 *    Plain white walls and matte boards produce almost no optical
 *    flow because there is nothing to track.  Edge density catches
 *    these cases by detecting structural complexity regardless of
 *    motion.  It remains as a fallback.
 *
 *  GATE DETECTION: detect_gate_yuv()  (UNCHANGED)
 *    The gate has a known, consistent colour (green).  YCbCr
 *    thresholding is still exactly the right tool here.
 *
 *  NAV STATE MACHINE: centre_safe logic updated
 *    Old: floor_pixels_center > threshold AND edge < threshold AND div < threshold
 *    New: divergence < threshold AND edge < threshold
 *         (floor pixels no longer involved in any navigation decision)
 *
 *  choose_turn_direction() updated
 *    Old: floor_pixels score minus edge penalty
 *    New: edge density only — pick the side with less structural clutter
 * ═══════════════════════════════════════════════════════════════════
 *
 * UYVY buffer layout (Bebop front camera, 856 × 480):
 *   Every 4-byte macro-pixel: [U=Cb  Y0  V=Cr  Y1]
 *   U (Cb) at buf[base+0], Y at buf[base+1 or +3], V (Cr) at buf[base+2]
 *
 * All processing is in native YUV space — no RGB or HSV anywhere.
 */

#include "cv_obstacle_avoidance.h"

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#include "firmwares/rotorcraft/guidance/guidance_h.h"
#include "autopilot.h"
#include "state.h"
#include "modules/core/abi.h"
#include "modules/computer_vision/cv.h"
#include "modules/computer_vision/lib/vision/image.h"

/* ═══════════════════════════════════════════════════════════════════
 * Default parameters  (override in airframe XML)
 * ═══════════════════════════════════════════════════════════════════ */

#ifndef OA_MAX_SPEED
#define OA_MAX_SPEED                  0.5f   /* m/s cruise                 */
#endif
#ifndef OA_HEADING_RATE
#define OA_HEADING_RATE               0.40f  /* rad per avoid step (~23°)  */
#endif
#ifndef OA_AVOID_DWELL_CYCLES
#define OA_AVOID_DWELL_CYCLES         25     /* cycles before first turn   */
#endif

/*
 * PRIMARY threshold – optical flow divergence.
 * Increase if drone stops too often in open space (false positives).
 * Decrease if drone gets too close before reacting.
 */
#ifndef OA_DIVERGENCE_THRESHOLD
#define OA_DIVERGENCE_THRESHOLD       0.035f
#endif

/*
 * Hysteresis threshold – must be BELOW this after avoiding before
 * resuming cruise.  Set lower than OA_DIVERGENCE_THRESHOLD to stop
 * the drone toggling rapidly between SAFE and AVOIDING.
 */
#ifndef OA_DIVERGENCE_CLEAR_THRESHOLD
#define OA_DIVERGENCE_CLEAR_THRESHOLD 0.018f
#endif

/*
 * SECONDARY threshold – Sobel edge density (edges per pixel).
 * Catches textureless obstacles that produce no flow.
 */
#ifndef OA_EDGE_DENSITY_THRESHOLD
#define OA_EDGE_DENSITY_THRESHOLD     0.25f
#endif

/* GREEN gate defaults – calibrate with YUVCalibrator */
#ifndef OA_GATE_Y_MIN
#define OA_GATE_Y_MIN    80
#define OA_GATE_Y_MAX    220
#define OA_GATE_CB_MIN   60
#define OA_GATE_CB_MAX   112
#define OA_GATE_CR_MIN   60
#define OA_GATE_CR_MAX   112
#endif

/* ═══════════════════════════════════════════════════════════════════
 * Exported tunables
 * ═══════════════════════════════════════════════════════════════════ */
float     oa_max_speed                  = OA_MAX_SPEED;
float     oa_heading_rate               = OA_HEADING_RATE;
uint16_t  oa_avoid_dwell_cycles         = OA_AVOID_DWELL_CYCLES;
float     oa_divergence_threshold       = OA_DIVERGENCE_THRESHOLD;
float     oa_divergence_clear_threshold = OA_DIVERGENCE_CLEAR_THRESHOLD;
float     oa_edge_density_threshold     = OA_EDGE_DENSITY_THRESHOLD;

uint8_t oa_gate_y_min  = OA_GATE_Y_MIN,  oa_gate_y_max  = OA_GATE_Y_MAX;
uint8_t oa_gate_cb_min = OA_GATE_CB_MIN, oa_gate_cb_max = OA_GATE_CB_MAX;
uint8_t oa_gate_cr_min = OA_GATE_CR_MIN, oa_gate_cr_max = OA_GATE_CR_MAX;

/* ═══════════════════════════════════════════════════════════════════
 * Internal state
 * ═══════════════════════════════════════════════════════════════════ */
volatile vision_results_t oa_vision;
volatile nav_state_t      oa_nav_state = NAV_SAFE;

static uint16_t  avoid_cycles_remaining = 0;
static float     current_heading_cmd    = 0.f;
static float     gate_approach_heading  = 0.f;
static uint8_t   heading_sign           = 1;   /* +1 = right, -1 = left  */
static uint16_t clear_count             = 0;
static uint16_t emergency_timer         = 0;
static uint16_t cruise_immunity         = 0;
static float accumulated_turn           = 0.f;
/* Previous frame for optical flow */
static struct image_t prev_frame;
static uint8_t        prev_frame_valid  = 0;
static uint16_t startup_dwell           = 0;
static float  best_scan_heading         = 0.f;
static float  best_scan_edge_c          = 1.0f;
static uint8_t   desperate_mode         = 0;




/* ═══════════════════════════════════════════════════════════════════
 * UYVY pixel access macros
 *
 *   Layout per 4-byte macro-pixel covering columns x and x+1:
 *     byte 0 = U  (Cb, shared)
 *     byte 1 = Y0 (luma for even column x)
 *     byte 2 = V  (Cr, shared)
 *     byte 3 = Y1 (luma for odd  column x+1)
 * ═══════════════════════════════════════════════════════════════════ */
#define UYVY_BASE(x, y, w)   ( ((y)*(w)+(x)) / 2 * 4 )
#define UYVY_Y(buf, x, y, w) ((buf)[UYVY_BASE(x,y,w) + (((x)%2==0) ? 1 : 3)])
#define UYVY_U(buf, x, y, w) ((buf)[UYVY_BASE(x,y,w) + 0])   /* Cb */
#define UYVY_V(buf, x, y, w) ((buf)[UYVY_BASE(x,y,w) + 2])   /* Cr */

/* ═══════════════════════════════════════════════════════════════════
 * 1.  OPTICAL FLOW DIVERGENCE  (PRIMARY obstacle detector)
 *
 *     Sparse 8×8 block-matching grid.  For each grid point:
 *       a) Find the best-matching 3×3 patch in the previous frame
 *          within a ±4 pixel search window (Y channel only).
 *       b) Project the resulting motion vector onto the radial
 *          direction (away from image centre).
 *       c) Divide by distance from centre to get divergence.
 *
 *     Per-third divergence is also computed so choose_turn_direction()
 *     can tell which side of the frame has the looming obstacle.
 *
 *     Why Y channel only:
 *       Luma carries all structural detail.  Using only Y halves the
 *       memory bandwidth compared to full YUV, which matters on the
 *       Bebop's ARM Cortex-A9.
 * ═══════════════════════════════════════════════════════════════════ */
static void compute_flow_divergence_y(const struct image_t *curr,
                                       const struct image_t *prev,
                                       float *div_total,
                                       float *div_left,
                                       float *div_center,
                                       float *div_right)
{
    *div_total = *div_left = *div_center = *div_right = 0.f;

    if (!prev || curr->w != prev->w || curr->h != prev->h) return;

    const uint32_t  w    = curr->w;
    const uint32_t  h    = curr->h;
    const uint8_t  *cbuf = (const uint8_t *)curr->buf;
    const uint8_t  *pbuf = (const uint8_t *)prev->buf;

    const float     cx   = w * 0.5f;
    const float     cy   = h * 0.5f;
    const uint32_t  stx  = w / 8;
    const uint32_t  sty  = h / 8;
    const uint32_t  col3 = w / 3;

    float sum_l = 0.f, sum_c = 0.f, sum_r = 0.f;
    int   cnt_l = 0,   cnt_c = 0,   cnt_r = 0;

    for (uint32_t gy = 1; gy < 7; gy++) {
        for (uint32_t gx = 1; gx < 7; gx++) {
            uint32_t px = gx * stx;
            uint32_t py = gy * sty;

            /* ── Block-matching: find best dx, dy within ±4 pixels ── */
            int16_t  best_dx = 0, best_dy = 0;
            uint32_t best_sad = UINT32_MAX;

            for (int16_t dy = -4; dy <= 4; dy++) {
                for (int16_t dx = -4; dx <= 4; dx++) {
                    uint32_t sad = 0;
                    for (int kx = -1; kx <= 1; kx++) {
                        for (int ky = -1; ky <= 1; ky++) {
                            int cx_ = (int)px + kx;
                            int cy_ = (int)py + ky;
                            int px_ = cx_ + dx;
                            int py_ = cy_ + dy;
                            if (py_ < 0 || py_ >= (int)h ||
                                px_ < 0 || px_ >= (int)w) {
                                sad += 255; continue;
                            }
                            sad += (uint32_t)abs(
                                (int)UYVY_Y(cbuf, cx_, cy_, w) -
                                (int)UYVY_Y(pbuf, px_, py_, w));
                        }
                    }
                    if (sad < best_sad) {
                        best_sad = sad;
                        best_dx  = dx;
                        best_dy  = dy;
                    }
                }
            }

            /* ── Radial divergence at this grid point ── */
            float rx   = (float)px - cx;
            float ry   = (float)py - cy;
            float r    = sqrtf(rx*rx + ry*ry) + 1e-6f;
            float rdiv = (best_dx*rx + best_dy*ry) / (r * r);

            /* Accumulate into the column third this point belongs to */
            if      (px < col3)       { sum_l += rdiv; cnt_l++; }
            else if (px < 2 * col3)   { sum_c += rdiv; cnt_c++; }
            else                      { sum_r += rdiv; cnt_r++; }
        }
    }

    *div_left   = (cnt_l > 0) ? sum_l / cnt_l : 0.f;
    *div_center = (cnt_c > 0) ? sum_c / cnt_c : 0.f;
    *div_right  = (cnt_r > 0) ? sum_r / cnt_r : 0.f;
    *div_total  = (*div_left + *div_center + *div_right) / 3.f;
}

/* ═══════════════════════════════════════════════════════════════════
 * 2.  SOBEL EDGE DENSITY  (SECONDARY obstacle detector)
 *
 *     Measures structural complexity per column third using the Y
 *     channel.  High edge density in the centre → likely obstacle
 *     even if there is no optical flow (textureless surface).
 *
 *     Only the upper 60 % of the frame is examined because the lower
 *     40 % shows the floor, which always has edges.
 * ═══════════════════════════════════════════════════════════════════ */
static void compute_edge_density_y(const struct image_t *img,
                                    float *left,
                                    float *center,
                                    float *right)
{
    const uint32_t  w       = img->w;
    const uint32_t  h       = img->h;
    const uint8_t  *buf     = (const uint8_t *)img->buf;
    const uint32_t  row_end = (uint32_t)(h * 0.60f);
    const uint32_t  col3    = w / 3;

    uint32_t el = 0, ec = 0, er = 0;
    uint32_t pl = 0, pc = 0, pr = 0;

    for (uint32_t y = 1; y < row_end - 1; y++) {
        for (uint32_t x = 1; x < w - 1; x++) {
            int16_t gx =
                (int16_t)UYVY_Y(buf,x+1,y-1,w) + 2*(int16_t)UYVY_Y(buf,x+1,y,w)
              + (int16_t)UYVY_Y(buf,x+1,y+1,w)
              - (int16_t)UYVY_Y(buf,x-1,y-1,w) - 2*(int16_t)UYVY_Y(buf,x-1,y,w)
              - (int16_t)UYVY_Y(buf,x-1,y+1,w);

            int16_t gy =
                (int16_t)UYVY_Y(buf,x-1,y+1,w) + 2*(int16_t)UYVY_Y(buf,x,y+1,w)
              + (int16_t)UYVY_Y(buf,x+1,y+1,w)
              - (int16_t)UYVY_Y(buf,x-1,y-1,w) - 2*(int16_t)UYVY_Y(buf,x,y-1,w)
              - (int16_t)UYVY_Y(buf,x+1,y-1,w);

            uint8_t edge = ((uint16_t)(abs(gx) + abs(gy)) > 80) ? 1 : 0;

            if      (x < col3)      { el += edge; pl++; }
            else if (x < 2 * col3)  { ec += edge; pc++; }
            else                    { er += edge; pr++; }
        }
    }

    *left   = (pl > 0) ? (float)el / pl : 0.f;
    *center = (pc > 0) ? (float)ec / pc : 0.f;
    *right  = (pr > 0) ? (float)er / pr : 0.f;
}

/* -----------------------------------------------------------------------
 * Checks average Y (luma) in the centre column third of the frame.
 * Returns 1 if the centre is abnormally dark — likely a black wall.
 * A black wall close-up fills the frame with low Y values but few edges.
 * --------------------------------------------------------------------- */
static uint8_t centre_is_dark(const struct image_t *img)
{
    const uint32_t  w       = img->w;
    const uint32_t  h       = img->h;
    const uint8_t  *buf     = (const uint8_t *)img->buf;
    const uint32_t  x_start = w / 3;
    const uint32_t  x_end   = 2 * w / 3;
    /* Only check middle rows — exclude floor and ceiling */
    const uint32_t  y_start = h / 4;
    const uint32_t  y_end   = 3 * h / 4;

    uint32_t sum   = 0;
    uint32_t count = 0;

    for (uint32_t y = y_start; y < y_end; y += 4) {   /* stride 4 for speed */
        for (uint32_t x = x_start; x < x_end; x += 4) {
            sum += UYVY_Y(buf, x, y, w);
            count++;
        }
    }

    if (count == 0) return 0;
    float mean_y = (float)sum / count;

    /* Below 40 luma = very dark = likely black wall close up */
    return (mean_y < 40.f) ? 1 : 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * 3.  YUV GATE DETECTOR  (colour blob – gate only)
 *
 *     Finds the largest blob matching the gate's YCbCr range.
 *     For a GREEN gate both Cb and Cr are below 128.
 *     Returns 1 on a valid detection, 0 otherwise.
 *
 *     Note: this is the ONLY place colour is used.  It is NOT called
 *     for obstacle detection — only for gate localisation.
 * ═══════════════════════════════════════════════════════════════════ */
static uint8_t detect_gate_yuv(const struct image_t *img,
                                float *gx_norm,
                                float *gy_norm,
                                float *gs_norm)
{
    const uint32_t  w   = img->w;
    const uint32_t  h   = img->h;
    const uint8_t  *buf = (const uint8_t *)img->buf;

    uint64_t sx = 0, sy = 0;
    uint32_t cnt = 0;
    uint32_t x_min = w, x_max = 0, y_min = h, y_max = 0;

    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            uint8_t Y  = UYVY_Y(buf, x, y, w);
            uint8_t Cb = UYVY_U(buf, x, y, w);
            uint8_t Cr = UYVY_V(buf, x, y, w);

            if (Y  >= oa_gate_y_min  && Y  <= oa_gate_y_max  &&
                Cb >= oa_gate_cb_min && Cb <= oa_gate_cb_max  &&
                Cr >= oa_gate_cr_min && Cr <= oa_gate_cr_max) {
                sx += x; sy += y; cnt++;
                if (x < x_min) x_min = x;
                if (x > x_max) x_max = x;
                if (y < y_min) y_min = y;
                if (y > y_max) y_max = y;
            }
        }
    }

    /* Require at least 0.8 % of frame and a plausible aspect ratio */
    if (cnt < (uint32_t)(w * h * 0.008f)) return 0;

    uint32_t bw = x_max - x_min;
    uint32_t bh = y_max - y_min;
    if (bh == 0) return 0;

    float aspect = (float)bw / bh;
    if (aspect < 0.30f || aspect > 3.50f) return 0;

    *gx_norm = ((float)(sx / cnt) / w) * 2.f - 1.f;
    *gy_norm = ((float)(sy / cnt) / h) * 2.f - 1.f;
    *gs_norm = (float)bw / w;
    return 1;
}

/* ═══════════════════════════════════════════════════════════════════
 * CV callback – executes in the Paparazzi vision thread
 * Called once per camera frame.
 * ═══════════════════════════════════════════════════════════════════ */
struct image_t *cv_oa_vision_cb(struct image_t *img, uint8_t camera_id)
{
    (void)camera_id;

    /* Temporary debug counter - remove after confirming frames arrive */

    static uint32_t frame_count = 0;
    frame_count++;
    if (frame_count % 30 == 0){
        printf("[OA] frames received: %u state=%d div=%.4f flow_c=%.4f" "edge_L=%.3f edge_c=%.3f edge_R=%.3f dark=%d\n",frame_count,(int)oa_nav_state,(float)oa_vision.divergence,(float)oa_vision.flow_center,(float)oa_vision.edge_density_left,(float)oa_vision.edge_density_center,(float)oa_vision.edge_density_right,(int)oa_vision.centre_dark);
    }

    /* ── 1. Optical flow divergence (PRIMARY obstacle signal) ─────── */
    float dt, dl, dc, dr;
    compute_flow_divergence_y(img,
                               prev_frame_valid ? &prev_frame : NULL,
                               &dt, &dl, &dc, &dr);
    oa_vision.divergence    = dt;
    oa_vision.flow_left     = dl;
    oa_vision.flow_center   = dc;
    oa_vision.flow_right    = dr;

    /* Store this frame as the "previous" for next callback */
    if (!prev_frame_valid) {
        image_create(&prev_frame, img->w, img->h, img->type);
        prev_frame_valid = 1;
    }
    image_copy(img, &prev_frame);

    /* ── 2. Edge density (SECONDARY obstacle signal) ──────────────── */
    float el, ec, er;
    compute_edge_density_y(img, &el, &ec, &er);
    oa_vision.edge_density_left   = el;
    oa_vision.edge_density_center = ec;
    oa_vision.edge_density_right  = er;

    oa_vision.centre_dark = centre_is_dark(img);
    /* ── 3. Gate detection (colour – only if offboard hasn't sent one) */
    if (!oa_vision.gate_detected) {
        float gx, gy, gs;
        if (detect_gate_yuv(img, &gx, &gy, &gs)) {
            oa_vision.gate_x_norm    = gx;
            oa_vision.gate_y_norm    = gy;
            oa_vision.gate_size_norm = gs;
            oa_vision.gate_confidence = 0.6f;  /* Onboard fallback confidence */
            oa_vision.gate_detected   = 1;
        }
    }

    return img;
}

/* ═══════════════════════════════════════════════════════════════════
 * IvyBus gate injection
 * Called when the offboard Python sends a gate detection.
 * Offboard detections take priority over onboard fallback.
 * ═══════════════════════════════════════════════════════════════════ */
void oa_set_gate_detection(float x_norm, float y_norm,
                            float size_norm, float confidence)
{
    oa_vision.gate_x_norm     = x_norm;
    oa_vision.gate_y_norm     = y_norm;
    oa_vision.gate_size_norm  = size_norm;
    oa_vision.gate_confidence = confidence;
    oa_vision.gate_detected   = 1;
}

/* ═══════════════════════════════════════════════════════════════════
 * choose_turn_direction()
 *
 * Returns the turn direction that gives the CLEAREST path.
 * Uses ONLY edge density — picks the side with less structural clutter.
 * (Floor pixels are no longer part of this decision.)
 *
 * Also factors in the per-third divergence: if one side already has
 * high divergence even at the periphery, it is likely more blocked.
 * ═══════════════════════════════════════════════════════════════════ */
static int8_t choose_turn_direction(void)
{
    /* Lower score = clearer path.  Weight divergence more than edges. */
    float score_left  = oa_vision.flow_left   * 2.0f
                      + oa_vision.edge_density_left;
    float score_right = oa_vision.flow_right  * 2.0f
                      + oa_vision.edge_density_right;

    return (score_left <= score_right) ? -1 : 1;  /* -1=left, +1=right */
}

/* ═══════════════════════════════════════════════════════════════════
 * path_is_clear()
 *
 * Returns 1 if the centre third of the frame is free of obstacles.
 *
 *   PRIMARY check : centre divergence below the CLEAR threshold
 *                   (uses hysteresis — slightly lower than trigger)
 *   SECONDARY check: centre edge density below threshold
 *
 * Both must pass.
 * ═══════════════════════════════════════════════════════════════════ */
static uint8_t path_is_clear(void)
{
    /* Only the centre matters — we are evaluating the forward path.
     * Both flow AND edge must be below their clear thresholds,
     * AND the centre must not be dark. */

    if(!desperate_mode){
        if(oa_vision.edge_density_center < 0.005f) return 0;
    }
    return (oa_vision.flow_center         < oa_divergence_clear_threshold)
        && (oa_vision.edge_density_center < oa_edge_density_threshold)
        && (!oa_vision.centre_dark);
}

/* ═══════════════════════════════════════════════════════════════════
 * obstacle_detected()
 *
 * Returns 1 if an obstacle is present in the centre of the frame.
 * Uses the TRIGGER threshold (higher than clear threshold).
 * ═══════════════════════════════════════════════════════════════════ */
static uint8_t obstacle_detected(void)
{
    /* Very close: high edge density alone is enough */
    if (oa_vision.edge_density_center > oa_edge_density_threshold * 2.0f)
        return 1;

    /* Mid-range: need edge AND some flow to confirm approach */
    if (oa_vision.edge_density_center > oa_edge_density_threshold &&
        oa_vision.flow_center > oa_divergence_threshold * 0.5f)
        return 1;

    /* Dark wall close-up */
    if (oa_vision.centre_dark) return 1;

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * Module init
 * ═══════════════════════════════════════════════════════════════════ */
void cv_obstacle_avoidance_init(void)
{
    


    memset((void *)&oa_vision, 0, sizeof(oa_vision));
    oa_nav_state        = NAV_SAFE;
    current_heading_cmd = stateGetNedToBodyEulers_f()->psi;

    cv_add_to_device(&front_camera, cv_oa_vision_cb, 0,0);

    printf("[OA] Initialised – FLOW+EDGE obstacles / YUV gate detection\n");
    printf("[OA] div_trigger=%.3f  div_clear=%.3f  edge=%.3f\n",
           oa_divergence_threshold,
           oa_divergence_clear_threshold,
           oa_edge_density_threshold);
}

/* ═══════════════════════════════════════════════════════════════════
 * Module periodic – navigation state machine  (@ 30 Hz)
 * ═══════════════════════════════════════════════════════════════════ */
void cv_obstacle_avoidance_periodic(void)
{
    if (autopilot_get_mode() != AP_MODE_GUIDED) return;
    
    if (startup_dwell < 300) {
        startup_dwell++;
        if (startup_dwell == 300) {
            current_heading_cmd = stateGetNedToBodyEulers_f()->psi;
        }
        return;
    }

    switch (oa_nav_state) {

    /* ── NAV_SAFE ──────────────────────────────────────────────────
     * Default cruising state.
     * Checks gate detection first (higher priority than obstacle).
     * ──────────────────────────────────────────────────────────── */
    case NAV_SAFE:
        if (oa_vision.gate_detected) {
            oa_nav_state    = NAV_GATE_FOUND;
            cruise_immunity = 0;
            break;
        }

    /* Immunity period — drone just found a clear heading and needs
     * to physically move away from the wall before re-evaluating.
     * During this window ignore obstacle signals and just cruise. */
        if (cruise_immunity > 0) {
            cruise_immunity--;
            guidance_h_set_heading(current_heading_cmd);
            guidance_h_set_body_vel(oa_max_speed * 0.35f, 0.f);

            /* Boundary check still active during immunity */
            {
                struct EnuCoor_f *pos = stateGetPositionEnu_f();
                float dist_sq = pos->x * pos->x + pos->y * pos->y;
                if (dist_sq > 20.25f) {
                    cruise_immunity        = 0;
                    avoid_cycles_remaining = oa_avoid_dwell_cycles;
                    oa_nav_state           = NAV_OBSTACLE_NEAR;
                    break;
                }
            }
    /* Emergency override — very close obstacle during immunity */
            if (obstacle_detected()) {
                cruise_immunity        = 0;
                avoid_cycles_remaining = oa_avoid_dwell_cycles;
                oa_nav_state           = NAV_OBSTACLE_NEAR;
            }
            break;
        }
        
        if(oa_vision.edge_density_right > oa_edge_density_threshold * 4.0f || 
           oa_vision.edge_density_left > oa_edge_density_threshold * 4.0f) {
            avoid_cycles_remaining = oa_avoid_dwell_cycles;
            oa_nav_state = NAV_OBSTACLE_NEAR;
            break;
           }

        if(obstacle_detected()){
            avoid_cycles_remaining = oa_avoid_dwell_cycles;
            oa_nav_state           = NAV_OBSTACLE_NEAR;
            break;
        }

    /* Boundary check */
        {
            struct EnuCoor_f *pos = stateGetPositionEnu_f();
            float dist_sq = pos->x * pos->x + pos->y * pos->y;
            if (dist_sq > 20.25f) {
                guidance_h_set_body_vel(0.f, 0.f);
                avoid_cycles_remaining = oa_avoid_dwell_cycles;
                oa_nav_state           = NAV_OBSTACLE_NEAR;
                break;
            }
        }

        guidance_h_set_heading(current_heading_cmd);
        guidance_h_set_body_vel(oa_max_speed, 0.f);
        break;

    /* ── NAV_OBSTACLE_NEAR ─────────────────────────────────────────
     * Brake to a full stop and wait oa_avoid_dwell_cycles frames
     * before deciding which way to turn.  Waiting lets the optical
     * flow settle (flow needs forward motion to be meaningful, so
     * we let the drone decelerate fully before scoring directions).
     * ──────────────────────────────────────────────────────────── */
    case NAV_OBSTACLE_NEAR:
        clear_count     = 0;
        emergency_timer = 0;
        cruise_immunity = 0;
        guidance_h_set_body_vel(0.f, 0.f);

        if (avoid_cycles_remaining > 0) {
            avoid_cycles_remaining--;
            break;
        }

    /*
     * Choose turn direction based on edge density,
     * then ADD a random component (0.2 to 1.2 radians extra)
     * so the drone explores a different heading each time
     * rather than returning to the same direction.
     */
        {
            float score_l = oa_vision.flow_left  + oa_vision.edge_density_left;
            float score_r = oa_vision.flow_right + oa_vision.edge_density_right;
            heading_sign  = (score_l <= score_r) ? -1 : 1;
            accumulated_turn = 0.f;
        }
        /* Sync heading to actual current heading before scanning */
        current_heading_cmd = stateGetNedToBodyEulers_f()->psi;
        accumulated_turn    = 0.f;
        best_scan_edge_c    = 1.0f;
        best_scan_heading   = current_heading_cmd;
        avoid_cycles_remaining = oa_avoid_dwell_cycles * 2;
        oa_nav_state           = NAV_AVOIDING;
        break;

    /* ── NAV_AVOIDING ──────────────────────────────────────────────
     * Rotate in the chosen direction and creep forward slowly.
     * Re-evaluate after oa_avoid_dwell_cycles * 2 frames.
     * If the centre clears, resume SAFE.
     * If a gate appears during avoidance, jump to GATE_FOUND.
     * After ~360° of accumulated turning with no clear path → EMERGENCY.
     * ──────────────────────────────────────────────────────────── */
    case NAV_AVOIDING:
        if (oa_vision.gate_detected) {
            oa_nav_state = NAV_GATE_FOUND;
            clear_count  = 0;
            break;
        }

    /*
     * Rotate slowly at 0.5 deg per cycle (0.009 rad).
     * Check every frame if the centre is clear.
     * The MOMENT 5 consecutive clear frames are seen,
     * lock the current heading and cruise in that direction.
     * This guarantees the drone commits to a heading it
     * has visually confirmed is open — not a random guess.
     */
        current_heading_cmd += heading_sign * 0.009f;
        accumulated_turn    += 0.009f;

    /* Wrap to [-π, +π] — without this heading grows to 447 rad */
        while (current_heading_cmd >  (float)M_PI) current_heading_cmd -= 2.f * (float)M_PI;
        while (current_heading_cmd < -(float)M_PI) current_heading_cmd += 2.f * (float)M_PI;
        if (oa_vision.edge_density_center > 0.010f &&
            oa_vision.edge_density_center < best_scan_edge_c) {
            best_scan_edge_c  = oa_vision.edge_density_center;
            best_scan_heading = current_heading_cmd;
        }
        guidance_h_set_heading(current_heading_cmd);
        guidance_h_set_body_vel(0.f, 0.f);   /* stand still while scanning */

        if (path_is_clear()) {
            clear_count++;
            if (clear_count >= 8) {
            /* Confirmed clear — lock this heading and cruise */
                desperate_mode = 0;
                clear_count  = 0;
                cruise_immunity = 40;
                oa_nav_state = NAV_SAFE;
                printf("[OA] Clear heading locked at %.2f rad\n", current_heading_cmd);
            }
        } else {
            clear_count = 0;
        }   

    /* Safety: more than 360 degrees scanned with nothing clear */
        {
            float psi  = stateGetNedToBodyEulers_f()->psi;
            float diff = fabsf(current_heading_cmd - psi);
            if (accumulated_turn > 6.45f) {
                oa_nav_state = NAV_EMERGENCY;
            }
        }
        break;

    /* ── NAV_GATE_FOUND ────────────────────────────────────────────
     * A gate has been detected.  Apply a proportional heading
     * correction to centre the gate in the frame, then wait until
     * the gate is both centred (|x_norm| < 0.15) and large enough
     * (size_norm > 0.10) before committing to an approach.
     *
     * Abort back to SAFE if the gate disappears.
     * ──────────────────────────────────────────────────────────── */
    case NAV_GATE_FOUND:
        if (!oa_vision.gate_detected) {
            oa_nav_state = NAV_SAFE;
            break;
        }
        {
            float err = oa_vision.gate_x_norm;
            float psi = stateGetNedToBodyEulers_f()->psi;

            /*
             * P-controller: heading correction proportional to how far
             * off-centre the gate is.  Gain 0.55 rad per unit x_norm.
             */
            gate_approach_heading = psi + err * 0.55f;
            guidance_h_set_heading(gate_approach_heading);
            guidance_h_set_body_vel(0.20f, 0.f);

            /* Ready to approach when aligned and close enough */
            if (fabsf(err) < 0.15f && oa_vision.gate_size_norm > 0.10f) {
                oa_nav_state = NAV_GATE_APPROACH;
            }
        }
        break;

    /* ── NAV_GATE_APPROACH ─────────────────────────────────────────
     * Aligned to gate – accelerate through.
     * Continue fine heading corrections using gate x_norm.
     * Trigger the PASS phase when gate fills >55 % of frame width.
     *
     * SAFETY: if strong divergence appears mid-approach (unexpected
     * obstacle between drone and gate), abort immediately.
     * ──────────────────────────────────────────────────────────── */
    case NAV_GATE_APPROACH:
        {
            float err = oa_vision.gate_detected ? oa_vision.gate_x_norm : 0.f;

            /* Fine correction at half the gain used during alignment */
            gate_approach_heading += err * 0.25f;
            guidance_h_set_heading(gate_approach_heading);
            guidance_h_set_body_vel(oa_max_speed * 1.30f, 0.f);

            /* Gate is filling the frame – go for it */
            if (oa_vision.gate_size_norm > 0.55f) {
                avoid_cycles_remaining = oa_avoid_dwell_cycles;
                oa_nav_state           = NAV_GATE_PASS;
                break;
            }

            /* Unexpected obstacle between drone and gate – abort */
            if (oa_vision.flow_center > oa_divergence_threshold * 1.8f) {
                avoid_cycles_remaining = oa_avoid_dwell_cycles;
                oa_nav_state           = NAV_OBSTACLE_NEAR;
            }
        }
        break;

    /* ── NAV_GATE_PASS ─────────────────────────────────────────────
     * Full-speed burst through the gate on the locked heading.
     * Hold for oa_avoid_dwell_cycles frames, then resume SAFE.
     * ──────────────────────────────────────────────────────────── */
    case NAV_GATE_PASS:
        guidance_h_set_heading(gate_approach_heading);
        guidance_h_set_body_vel(oa_max_speed * 1.6f, 0.f);
        oa_vision.gate_detected = 0;   /* Clear – look for next gate */

        if (avoid_cycles_remaining > 0) {
            avoid_cycles_remaining--;
        } else {
            oa_nav_state = NAV_SAFE;
        }
        break;

    /* ── NAV_EMERGENCY ─────────────────────────────────────────────
     * All headings tried, nothing clear.  Hover and wait.
     * If the path clears by itself (obstacle moves), resume SAFE.
     * ──────────────────────────────────────────────────────────── */
    case NAV_EMERGENCY:
        {
            guidance_h_set_body_vel(-0.35f, 0.f);
            emergency_timer++;

        /* Try path_is_clear first */
            if (path_is_clear()) {
                emergency_timer = 0;
                oa_nav_state    = NAV_SAFE;
                break;
            }

        /* After 3 seconds (90 cycles @ 30Hz), force a reset.
         * The drone is genuinely cornered — pick a random 
         * large turn and try again rather than hovering forever. */
            if (emergency_timer > 60) {
                emergency_timer        = 0;
                desperate_mode         = 1;
                current_heading_cmd    = best_scan_heading;
                best_scan_edge_c       = 1.0f;
                cruise_immunity        = 20;
                clear_count            = 0;
                accumulated_turn       = 0.f;
                heading_sign           = (heading_sign == 1) ? -1 : 1;
                oa_nav_state           = NAV_SAFE;
            }
        }
        break;
    }
}

/*
 * ═══════════════════════════════════════════════════════════════════════
 * AIRFRAME XML SNIPPET  (add to your bebop2.xml)
 * ═══════════════════════════════════════════════════════════════════════
 *
 * <modules>
 *   <module name="cv_obstacle_avoidance" dir="cv_oa">
 *
 *     <!-- Speeds -->
 *     <define name="OA_MAX_SPEED"                  value="0.5"/>
 *     <define name="OA_HEADING_RATE"               value="0.40"/>
 *     <define name="OA_AVOID_DWELL_CYCLES"         value="25"/>
 *
 *     <!-- Obstacle thresholds (tune with on-site test flights) -->
 *     <define name="OA_DIVERGENCE_THRESHOLD"       value="0.035"/>
 *     <define name="OA_DIVERGENCE_CLEAR_THRESHOLD" value="0.018"/>
 *     <define name="OA_EDGE_DENSITY_THRESHOLD"     value="0.25"/>
 *
 *     <!-- GREEN gate YCbCr – calibrate with YUVCalibrator -->
 *     <define name="OA_GATE_Y_MIN"   value="80"/>
 *     <define name="OA_GATE_Y_MAX"   value="220"/>
 *     <define name="OA_GATE_CB_MIN"  value="60"/>
 *     <define name="OA_GATE_CB_MAX"  value="112"/>
 *     <define name="OA_GATE_CR_MIN"  value="60"/>
 *     <define name="OA_GATE_CR_MAX"  value="112"/>
 *
 *   </module>
 * </modules>
 *
 * <section name="GUIDED_CONTROL">
 *   <define name="GUIDANCE_H_USE_REF"  value="FALSE"/>
 *   <define name="GUIDANCE_H_MAX_BANK" value="30" unit="deg"/>
 * </section>
 * ═══════════════════════════════════════════════════════════════════════
 */
