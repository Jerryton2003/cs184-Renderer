#ifndef CGL_CAMERA_H
#define CGL_CAMERA_H

#include <iostream>

#include "../util/device-vec-ops.h"
#include "CGL/matrix3x3.h"

#include "math.h"


namespace CGL {

/**
 * Camera.
 */
class Camera {
 public:
  /*
    Phi and theta are in RADIANS.
  */
  void place(const Vector3D targetPos, const double phi, const double theta,
             const double r, const double minR, const double maxR);

  Camera() = default;
  /*
    Copies just placement data from the other camera.
  */
  void copy_placement(const Camera &other);

  /*
    Updates the screen size to be the specified size, keeping screenDist
    constant.
  */
  void set_screen_size(const size_t screenW, const size_t screenH);

  /*
    Translates the camera such that a value at distance d directly in front of
    the camera moves by (dx, dy). Note that dx and dy are in screen coordinates,
    while d is in world-space coordinates (like pos/dir/up).
  */
  void move_by(const double dx, const double dy, const double d);

  Camera(const Camera& other) {
    hFov = other.hFov;
    vFov = other.vFov;
    ar = other.ar;
    nClip = other.nClip;
    fClip = other.fClip;
    lensRadius = other.lensRadius;
    focalDistance = other.focalDistance;
    pos = other.pos;
    targetPos = other.targetPos;
    phi = other.phi;
    theta = other.theta;
    r = other.r;
    minR = other.minR;
    maxR = other.maxR;
    c2w = other.c2w;
    screenW = other.screenW;
    screenH = other.screenH;
    screenDist = other.screenDist;
  }

  Camera& operator=(Camera&& other) noexcept {
    if (this != &other) {
      hFov = other.hFov;
      vFov = other.vFov;
      ar = other.ar;
      nClip = other.nClip;
      fClip = other.fClip;
      lensRadius = other.lensRadius;
      focalDistance = other.focalDistance;
      pos = other.pos;
      targetPos = other.targetPos;
      phi = other.phi;
      theta = other.theta;
      r = other.r;
      minR = other.minR;
      maxR = other.maxR;
      c2w = other.c2w;
      screenW = other.screenW;
      screenH = other.screenH;
      screenDist = other.screenDist;
    }
    return *this;
  }
  /*
    Move the specified amount along the view axis.
  */
  void move_forward(const double dist);

  /*
    Rotate by the specified amount around the target.
  */
  void rotate_by(const double dPhi, const double dTheta);

  double3 position() const { return pos; }

  double3 view_point() const { return targetPos; }

  double3 up_dir() const { return c2w[1]; }

  double v_fov() const { return vFov; }

  double aspect_ratio() const { return ar; }

  double near_clip() const { return nClip; }

  double far_clip() const { return fClip; }

  void dump_settings(std::string filename) {
    std::cerr << "Camera::dump_settings not implemented" << std::endl;
  }

  void load_settings(std::string filename) {
    std::cerr << "Camera::load_settings not implemented" << std::endl;
  }

  // Lens aperture and focal distance for depth of field effects.
  double lensRadius;
  double focalDistance;

  ~Camera() = default;

  // Computes pos, screenXDir, screenYDir from target, r, phi, theta.
  void compute_position();

  // Field of view aspect ratio, clipping planes.
  double hFov, vFov, ar, nClip, fClip;

  // Current position and target point (the point the camera is looking at).
  double3 pos, targetPos;

  // Orientation relative to target, and min & max distance from the target.
  double phi, theta, r, minR, maxR;

  // camera-to-world rotation matrix (note: also need to translate a
  // camera-space point by 'pos' to perform a full camera-to-world
  // transform)
  Mat3 c2w;

  // Info about screen to render to; it corresponds to the camera's full field
  // of view at some distance.
  size_t screenW, screenH;
  double screenDist;
  int8_t medium_id = -1;
};

} // namespace CGL

#endif // CGL_CAMERA_H
