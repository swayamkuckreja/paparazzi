/*
 * Copyright (C) 2019 Kirk Scheper <kirkscheper@gmail.com>
 *
 * This file is part of Paparazzi.
 *
 * Paparazzi is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 *
 * Paparazzi is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Paparazzi; see the file COPYING.  If not, write to
 * the Free Software Foundation, 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

/**
 * @file modules/computer_vision/cv_detect_object.h
 * Assumes the object consists of a continuous color and checks
 * if you are over the defined object or not
 */

 
#include "modules/computer_vision/cv_detect_color_object.h"
#include "modules/computer_vision/cv.h"
#include "modules/core/abi.h"
#include "modules/computer_vision/lib/vision/image.h"
#include "std.h"

#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include <pthread.h>

#ifndef OBJECT_DETECTOR_VERBOSE
#define OBJECT_DETECTOR_VERBOSE 0
#endif

#define FUNCTION __FUNCTION__
#define PRINT(fmt, ...) fprintf(stderr, "[cv_detect_color_object->%s()] " fmt, FUNCTION, ##__VA_ARGS__)
#if OBJECT_DETECTOR_VERBOSE
#define VERBOSE_PRINT(...) PRINT(__VA_ARGS__)
#else
#define VERBOSE_PRINT(...)
#endif

static pthread_mutex_t mutex;

#ifndef COLOR_OBJECT_DETECTOR_FPS1
#define COLOR_OBJECT_DETECTOR_FPS1 0
#endif
#ifndef COLOR_OBJECT_DETECTOR_FPS2
#define COLOR_OBJECT_DETECTOR_FPS2 0
#endif
#ifndef COLOR_OBJECT_DETECTOR_ROI_WIDTH_FRAC
#define COLOR_OBJECT_DETECTOR_ROI_WIDTH_FRAC 0.3f
#endif
#ifndef COLOR_OBJECT_DETECTOR_ROI_HEIGHT_FRAC
#define COLOR_OBJECT_DETECTOR_ROI_HEIGHT_FRAC 0.3f
#endif

/* Filter Settings (set via airframe defines + dl_settings) */
uint8_t cod_lum_min1 = 0, cod_lum_max1 = 0, cod_cb_min1 = 0, cod_cb_max1 = 0, cod_cr_min1 = 0, cod_cr_max1 = 0;
uint8_t cod_lum_min2 = 0, cod_lum_max2 = 0, cod_cb_min2 = 0, cod_cb_max2 = 0, cod_cr_min2 = 0, cod_cr_max2 = 0;
bool cod_draw1 = false;
bool cod_draw2 = false;

struct color_object_t {
  uint32_t roi_color_count;
  uint32_t roi_area;
  uint32_t roi2_count;
  uint32_t roi2_area;
  bool updated;
};
static struct color_object_t global_filters[2];

static void color_object_filter(struct image_t *img, bool draw,
                                uint8_t lum_min, uint8_t lum_max,
                                uint8_t cb_min,  uint8_t cb_max,
                                uint8_t cr_min,  uint8_t cr_max,
                                uint32_t *p_roi_color_count, uint32_t *p_roi_area,
                                uint32_t *p_roi2_color_count, uint32_t *p_roi2_area)
{
  uint8_t *buffer = (uint8_t *)img->buf;

  /* ROI1 = bottom-center (ground check)
   * ROI2 = top-center (tree/overhang check)
   */
  uint16_t roi_w  = (uint16_t)fmaxf(1.f, img->w * COLOR_OBJECT_DETECTOR_ROI_WIDTH_FRAC);
  uint16_t roi_h  = (uint16_t)fmaxf(1.f, img->h * COLOR_OBJECT_DETECTOR_ROI_HEIGHT_FRAC);
  uint16_t roi2_w = roi_w;
  uint16_t roi2_h = roi_h;

  if (roi_w > img->w) roi_w = img->w;
  if (roi_h > img->h) roi_h = img->h;
  if (roi2_w > img->w) roi2_w = img->w;
  if (roi2_h > img->h) roi2_h = img->h;

  uint16_t roi_x_min  = (img->w - roi_w) / 2;
  uint16_t roi_x_max  = roi_x_min + roi_w;
  uint16_t roi_y_min  = img->h - roi_h;
  uint16_t roi_y_max  = img->h;

  uint16_t roi2_x_min = (img->w - roi2_w) / 2;
  uint16_t roi2_x_max = roi2_x_min + roi2_w;
  uint16_t roi2_y_min = 0;
  uint16_t roi2_y_max = roi2_h;

  uint32_t roi_color_count = 0;
  uint32_t roi2_color_count = 0;

  if (draw) {
    struct point_t a, b;
    uint8_t c[4] = {90, 255, 240, 255};

    /* ROI1 box */
    a.x = roi_x_min; a.y = roi_y_min; b.x = roi_x_max; b.y = roi_y_min;
    image_draw_line_color(img, &a, &b, c);
    a.x = roi_x_min; a.y = roi_y_max; b.x = roi_x_max; b.y = roi_y_max;
    image_draw_line_color(img, &a, &b, c);
    a.x = roi_x_min; a.y = roi_y_min; b.x = roi_x_min; b.y = roi_y_max;
    image_draw_line_color(img, &a, &b, c);
    a.x = roi_x_max; a.y = roi_y_min; b.x = roi_x_max; b.y = roi_y_max;
    image_draw_line_color(img, &a, &b, c);

    /* ROI2 box */
    a.x = roi2_x_min; a.y = roi2_y_min; b.x = roi2_x_max; b.y = roi2_y_min;
    image_draw_line_color(img, &a, &b, c);
    a.x = roi2_x_min; a.y = roi2_y_max; b.x = roi2_x_max; b.y = roi2_y_max;
    image_draw_line_color(img, &a, &b, c);
    a.x = roi2_x_min; a.y = roi2_y_min; b.x = roi2_x_min; b.y = roi2_y_max;
    image_draw_line_color(img, &a, &b, c);
    a.x = roi2_x_max; a.y = roi2_y_min; b.x = roi2_x_max; b.y = roi2_y_max;
    image_draw_line_color(img, &a, &b, c);
  }

  /* iterate YUV422 (UYVY) */
  for (uint16_t y = 0; y < img->h; y++) {
    for (uint16_t x = 0; x < img->w; x++) {

      uint8_t *yp, *up, *vp;

      if ((x & 1) == 0) {
        /* even pixel: U Y0 V Y1 */
        up = &buffer[y * 2 * img->w + 2 * x + 0];
        yp = &buffer[y * 2 * img->w + 2 * x + 1];
        vp = &buffer[y * 2 * img->w + 2 * x + 2];
      } else {
        /* odd pixel shares U,V with previous even */
        up = &buffer[y * 2 * img->w + 2 * (x - 1) + 0];
        vp = &buffer[y * 2 * img->w + 2 * (x - 1) + 2];
        yp = &buffer[y * 2 * img->w + 2 * x + 1];
      }

      if ((*yp >= lum_min) && (*yp <= lum_max) &&
          (*up >= cb_min)  && (*up <= cb_max)  &&
          (*vp >= cr_min)  && (*vp <= cr_max)) {

        const bool in_roi1 = (x >= roi_x_min && x < roi_x_max && y >= roi_y_min && y < roi_y_max);
        const bool in_roi2 = (x >= roi2_x_min && x < roi2_x_max && y >= roi2_y_min && y < roi2_y_max);

        if (in_roi1) { roi_color_count++; }
        if (in_roi2) { roi2_color_count++; }

        if (draw) {
          *yp = 255;
        }
      }
    }
  }

  if (p_roi_color_count) *p_roi_color_count = roi_color_count;
  if (p_roi2_color_count) *p_roi2_color_count = roi2_color_count;
  if (p_roi_area) *p_roi_area = (uint32_t)roi_w * (uint32_t)roi_h;
  if (p_roi2_area) *p_roi2_area = (uint32_t)roi2_w * (uint32_t)roi2_h;
}

static struct image_t *object_detector(struct image_t *img, uint8_t filter)
{
  uint8_t lum_min, lum_max, cb_min, cb_max, cr_min, cr_max;
  bool draw;

  switch (filter) {
    case 1:
      lum_min = cod_lum_min1; lum_max = cod_lum_max1;
      cb_min  = cod_cb_min1;  cb_max  = cod_cb_max1;
      cr_min  = cod_cr_min1;  cr_max  = cod_cr_max1;
      draw = cod_draw1;
      break;
    case 2:
      lum_min = cod_lum_min2; lum_max = cod_lum_max2;
      cb_min  = cod_cb_min2;  cb_max  = cod_cb_max2;
      cr_min  = cod_cr_min2;  cr_max  = cod_cr_max2;
      draw = cod_draw2;
      break;
    default:
      return img;
  }

  uint32_t roi_c = 0, roi_a = 0, roi2_c = 0, roi2_a = 0;
  color_object_filter(img, draw, lum_min, lum_max, cb_min, cb_max, cr_min, cr_max,
                     &roi_c, &roi_a, &roi2_c, &roi2_a);

  pthread_mutex_lock(&mutex);
  global_filters[filter - 1].roi_color_count = roi_c;
  global_filters[filter - 1].roi_area       = roi_a;
  global_filters[filter - 1].roi2_count     = roi2_c;
  global_filters[filter - 1].roi2_area      = roi2_a;
  global_filters[filter - 1].updated        = true;
  pthread_mutex_unlock(&mutex);

  return img;
}

static struct image_t *object_detector1(struct image_t *img, uint8_t cam_id)
{
  (void)cam_id;
  return object_detector(img, 1);
}

static struct image_t *object_detector2(struct image_t *img, uint8_t cam_id)
{
  (void)cam_id;
  return object_detector(img, 2);
}

void color_object_detector_init(void)
{
  memset(global_filters, 0, sizeof(global_filters));
  pthread_mutex_init(&mutex, NULL);

#ifdef COLOR_OBJECT_DETECTOR_CAMERA1
#ifdef COLOR_OBJECT_DETECTOR_LUM_MIN1
  cod_lum_min1 = COLOR_OBJECT_DETECTOR_LUM_MIN1;
  cod_lum_max1 = COLOR_OBJECT_DETECTOR_LUM_MAX1;
  cod_cb_min1  = COLOR_OBJECT_DETECTOR_CB_MIN1;
  cod_cb_max1  = COLOR_OBJECT_DETECTOR_CB_MAX1;
  cod_cr_min1  = COLOR_OBJECT_DETECTOR_CR_MIN1;
  cod_cr_max1  = COLOR_OBJECT_DETECTOR_CR_MAX1;
#endif
#ifdef COLOR_OBJECT_DETECTOR_DRAW1
  cod_draw1 = COLOR_OBJECT_DETECTOR_DRAW1;
#endif
  cv_add_to_device(&COLOR_OBJECT_DETECTOR_CAMERA1, object_detector1, COLOR_OBJECT_DETECTOR_FPS1, 0);
#endif

#ifdef COLOR_OBJECT_DETECTOR_CAMERA2
#ifdef COLOR_OBJECT_DETECTOR_LUM_MIN2
  cod_lum_min2 = COLOR_OBJECT_DETECTOR_LUM_MIN2;
  cod_lum_max2 = COLOR_OBJECT_DETECTOR_LUM_MAX2;
  cod_cb_min2  = COLOR_OBJECT_DETECTOR_CB_MIN2;
  cod_cb_max2  = COLOR_OBJECT_DETECTOR_CB_MAX2;
  cod_cr_min2  = COLOR_OBJECT_DETECTOR_CR_MIN2;
  cod_cr_max2  = COLOR_OBJECT_DETECTOR_CR_MAX2;
#endif
#ifdef COLOR_OBJECT_DETECTOR_DRAW2
  cod_draw2 = COLOR_OBJECT_DETECTOR_DRAW2;
#endif
  cv_add_to_device(&COLOR_OBJECT_DETECTOR_CAMERA2, object_detector2, COLOR_OBJECT_DETECTOR_FPS2, 1);
#endif
}

void color_object_detector_periodic(void)
{
  static struct color_object_t local_filters[2];

  pthread_mutex_lock(&mutex);
  memcpy(local_filters, global_filters, sizeof(local_filters));
  pthread_mutex_unlock(&mutex);

  if (local_filters[0].updated) {
    int16_t roi_c  = (int16_t)Min(local_filters[0].roi_color_count, (uint32_t)INT16_MAX);
    int16_t roi_a  = (int16_t)Min(local_filters[0].roi_area,       (uint32_t)INT16_MAX);
    int32_t roi2_c = (int32_t)Min(local_filters[0].roi2_count,      (uint32_t)INT32_MAX);
    int16_t roi2_a = (int16_t)Min(local_filters[0].roi2_area,       (uint32_t)INT16_MAX);

    AbiSendMsgVISUAL_DETECTION(COLOR_OBJECT_DETECTION1_ID, 0, 0, roi_c, roi_a, roi2_c, roi2_a);
    local_filters[0].updated = false;
  }

  if (local_filters[1].updated) {
    int16_t roi_c  = (int16_t)Min(local_filters[1].roi_color_count, (uint32_t)INT16_MAX);
    int16_t roi_a  = (int16_t)Min(local_filters[1].roi_area,       (uint32_t)INT16_MAX);
    int32_t roi2_c = (int32_t)Min(local_filters[1].roi2_count,      (uint32_t)INT32_MAX);
    int16_t roi2_a = (int16_t)Min(local_filters[1].roi2_area,       (uint32_t)INT16_MAX);

    AbiSendMsgVISUAL_DETECTION(COLOR_OBJECT_DETECTION2_ID, 0, 0, roi_c, roi_a, roi2_c, roi2_a);
    local_filters[1].updated = false;
  }
}
