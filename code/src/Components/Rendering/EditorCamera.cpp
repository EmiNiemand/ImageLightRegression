#include "Components/Rendering/EditorCamera.h"
#include "Managers/InputManager.h"
#include "Core/Object.h"
#include "Components/Transform.h"
#include "CUM.h"
#include "Application.h"
#include "Macros.h"

EditorCamera::EditorCamera(Object *parent, int id) : Camera(parent, id) {
    enabled = false;
}

EditorCamera::~EditorCamera() = default;

void EditorCamera::OnCreate() {
    Camera::OnCreate();

    if (!editorCamera) {
        enabled = true;
        editorCamera = parent;
        activeCamera = parent;
    }
}

void EditorCamera::Update() {
    if (!enabled) return;

    if (InputManager::GetInstance()->IsKeyPressed(Key::KEY_LEFT_CONTROL) && InputManager::GetInstance()->IsKeyDown(Key::KEY_KP_0)) {
        Camera::ChangeActiveCamera();
    }

    Camera::Update();

    if (activeCamera != parent) return;

    InputManager* inputManager = InputManager::GetInstance();
    Transform* transform = parent->transform;

    Application* application = Application::GetInstance();
    Viewport* viewport = &Application::viewports[0];
    double cursorX, cursorY;
    glfwGetCursorPos(application->window, &cursorX, &cursorY);

    if (inputManager->IsKeyDown(Key::MOUSE_RIGHT_BUTTON) && CUM::IsInViewport(glm::ivec2(cursorX, cursorY), viewport)) {
        cursorPreviousX = cursorX;
        cursorPreviousY = cursorY;
    }

    if (inputManager->IsKeyPressed(Key::MOUSE_RIGHT_BUTTON) && cursorPreviousX > 0 && cursorPreviousY > 0) {
        glfwSetInputMode(application->window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        double mouseShiftX = cursorX - cursorPreviousX;
        double mouseShiftY = cursorY - cursorPreviousY;

        float yaw;
        float pitch;

        if (mouseShiftX != 0.0) {
            int direction = (int)mouseShiftX/abs((int)mouseShiftX);
            yaw = (float)(direction);
        }
        else {
            yaw = 0.0f;
        }
        if (mouseShiftY != 0.0) {
            int direction = (int)mouseShiftY/abs((int)mouseShiftY);
            pitch = (float)(direction);
        }
        else {
            pitch = 0.0f;
        }

        transform->SetLocalRotation(transform->GetLocalRotation() + glm::vec3(-pitch, -yaw, 0.0f) * rotationSpeed);


        if (inputManager->IsKeyPressed(Key::KEY_W)) {
            transform->SetLocalPosition(transform->GetLocalPosition() + transform->GetForward() * speed);
        }
        else if (inputManager->IsKeyPressed(Key::KEY_S)) {
            transform->SetLocalPosition(transform->GetLocalPosition() - transform->GetForward() * speed);
        }
        else if (inputManager->IsKeyPressed(Key::KEY_D)) {
            transform->SetLocalPosition(transform->GetLocalPosition() + transform->GetRight() * speed);
        }
        else if (inputManager->IsKeyPressed(Key::KEY_A)) {
            transform->SetLocalPosition(transform->GetLocalPosition() - transform->GetRight() * speed);
        }

        glfwSetCursorPos(application->window, cursorPreviousX, cursorPreviousY);
    }
    else if (inputManager->IsKeyReleased(Key::MOUSE_RIGHT_BUTTON)) {
        glfwSetInputMode(Application::GetInstance()->window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        cursorPreviousX = -1.0f;
        cursorPreviousY = -1.0f;
    }

    if (inputManager->IsKeyPressed(Key::KEY_LEFT_CONTROL) && inputManager->IsKeyDown(Key::KEY_KP_5)) {
        renderingCamera->transform->SetLocalPosition(parent->transform->GetGlobalPosition());
        renderingCamera->transform->SetLocalRotation(parent->transform->GetGlobalRotation());
    }
}

void EditorCamera::Save(nlohmann::json &json) {
    Camera::Save(json);

    json["ComponentType"] = "EditorCamera";
}

void EditorCamera::Load(nlohmann::json &json) {
    Camera::Load(json);
}
