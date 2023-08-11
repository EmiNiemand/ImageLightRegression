#ifndef IMAGELIGHTREGRESSION_EDITORCAMERA_H
#define IMAGELIGHTREGRESSION_EDITORCAMERA_H

#include "Components/Rendering/Camera.h"
#include "GLFW/glfw3.h"

class EditorCamera : public Camera {
private:
    double cursorPreviousX = -1.0f;
    double cursorPreviousY = -1.0f;

public:
    float speed = 0.01f;
    float rotationSpeed = 0.05f;

public:
    EditorCamera(Object *parent, int id);
    ~EditorCamera() override;

    void Update() override;
};


#endif //IMAGELIGHTREGRESSION_EDITORCAMERA_H
