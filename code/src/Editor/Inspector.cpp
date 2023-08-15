#include "Editor/Inspector.h"

void Inspector::ShowComponentProperties(const std::string &componentName) {
    if (componentName == "Transform") {
        ShowTransform();
    }
    else if (componentName == "Camera" || componentName == "EditorCamera") {
        ShowCamera();
    }
    else if (componentName == "Renderer") {
        ShowRenderer();
    }
    else if (componentName == "DirectionalLight") {
        ShowDirectionalLight();
    }
    else if (componentName == "PointLight") {
        ShowPointLight();
    }
    else if (componentName == "SpotLight") {
        ShowSpotLight();
    }
}

void Inspector::ShowTransform() {

}

void Inspector::ShowCamera() {

}

void Inspector::ShowRenderer() {

}

void Inspector::ShowPointLight() {

}

void Inspector::ShowSpotLight() {

}

void Inspector::ShowDirectionalLight() {

}
