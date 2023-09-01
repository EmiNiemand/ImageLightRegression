#ifndef IMAGELIGHTREGRESSION_CAMERA_H
#define IMAGELIGHTREGRESSION_CAMERA_H

#include "Components/Component.h"

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

class Camera : public Component {
protected:
    inline static Object* activeCamera;
    inline static Object* renderingCamera;
    inline static Object* editorCamera;
    inline static Object* previouslyActiveCamera;

    float fov = 45.0f;
    float zNear = 0.1f;
    float zFar = 100.0f;

public:
    Camera(Object* parent, int id);
    ~Camera() override;

    void OnCreate() override;
    void OnDestroy() override;
    void OnUpdate() override;

    void SetFOV(float inFOV);
    void SetZNear(float inZNear);
    void SetZFar(float inZFar);

    [[nodiscard]] float GetFOV() const;
    [[nodiscard]] float GetZNear() const;
    [[nodiscard]] float GetZFar() const;

    static void ChangeActiveCamera();
    static void SetActiveCamera(Object* inCamera);
    static void SetRenderingCamera(Object* inCameraObject);

    static Object* GetActiveCamera();
    static Object* GetEditorCamera();
    static Object* GetRenderingCamera();
    static Object* GetPreviouslyActiveCamera();


    [[nodiscard]] glm::mat4 GetViewMatrix() const;
    [[nodiscard]] glm::mat4 GetProjectionMatrix() const;

    void Save(nlohmann::json &json) override;
    void Load(nlohmann::json &json) override;
};


#endif //IMAGELIGHTREGRESSION_CAMERA_H
