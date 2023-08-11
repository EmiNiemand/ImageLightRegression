#ifndef IMAGELIGHTREGRESSION_CAMERA_H
#define IMAGELIGHTREGRESSION_CAMERA_H

#include "Components/Component.h"

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

class Camera : public Component {
protected:
    inline static Object* activeCamera;

    float fov = 45.0f;
    float zNear = 0.1f;
    float zFar = 100.0f;

public:
    Camera(Object *parent, int id);
    ~Camera() override;

    void OnUpdate() override;

    void SetFOV(float inFOV);
    void SetZNear(float inZNear);
    void SetZFar(float inZFar);
    static void SetActiveCamera(Object* inCameraObject);

    static Object* GetActiveCamera();
    [[nodiscard]] glm::mat4 GetViewMatrix() const;
    [[nodiscard]] glm::mat4 GetProjectionMatrix() const;
};


#endif //IMAGELIGHTREGRESSION_CAMERA_H