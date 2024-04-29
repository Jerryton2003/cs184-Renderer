#ifndef CGL_GLSCENE_DIRECTIONALLIGHT_H
#define CGL_GLSCENE_DIRECTIONALLIGHT_H

#include "scene.h"
#include "scene/light.h"

#include "application/visual_debugger.h"

namespace CGL { namespace GLScene {

class DirectionalLight : public SceneLight {
 public:

  DirectionalLight(const Collada::LightInfo& light_info, 
                   const Matrix4x4& transform) {
    this->spectrum = light_info.spectrum;
    this->direction = -(transform * Vector4D(light_info.direction, 1)).to3D();
    this->direction.normalize();
  }

  SceneObjects::SceneLight *get_static_light() const {
    SceneObjects::DirectionalLight* l =
      new SceneObjects::DirectionalLight(spectrum, direction);
    return l;
  }

  void render_debugger_node() {
    if (ImGui::TreeNode(this, "Directional Light 0x%x", this))
    {
      DragDouble3("Radiance", &spectrum[0], 0.005);
      DragDouble3("Direction", &direction[0], 0.005);
      ImGui::TreePop();
    }
  }

 private:

  Vector3D spectrum;
  Vector3D direction;

};

} // namespace GLScene
} // namespace CGL

#endif //CGL_GLSCENE_DIRECTIONALLIGHT_H
