#ifndef IMAGELIGHTREGRESSION_INSPECTOR_H
#define IMAGELIGHTREGRESSION_INSPECTOR_H

#include "glm/glm.hpp"
#include "imgui.h"

#include <string>

class Component;

class Inspector {
private:
    inline static glm::vec3 copiedVec3 = glm::vec3(0.0f);
    inline static glm::vec2 copiedVec2 = glm::vec3(0.0f);

public:
    static void ShowPopUp();
    static void ShowName();
    static void ShowComponentProperties(Component* component);

private:
    static void ShowTransform();
    static void ShowCamera();
    static void ShowRenderer();
    static void ShowDirectionalLight();
    static void ShowPointLight();
    static void ShowSpotLight();
    static void ShowSkybox();

    /// Returns whether a header is open or not
    static bool ShowComponentHeader(Component* component, const std::string& headerText);
    static bool ShowVec3(const char* label, glm::vec3& vec3, float speed = 0.1f, float resetValue = 0.0f,
                         float minValue = FLT_MAX, float maxValue = FLT_MAX);
    static bool ShowVec2(const char* label, glm::vec2& vec2, float speed = 0.1f, float resetValue = 0.0f,
                         float minValue = FLT_MAX, float maxValue = FLT_MAX);
    static bool ShowFloat(const char* label, float* value, float speed = 0.1f, float resetValue = 0.0f,
                          float minValue = FLT_MAX, float maxValue = FLT_MAX);
};


#endif //IMAGELIGHTREGRESSION_INSPECTOR_H
