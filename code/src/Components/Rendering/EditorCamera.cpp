#include "Components/Rendering/EditorCamera.h"
#include "Application.h"
#include "Managers/InputManager.h"
#include "Core/Object.h"
#include "Components/Transform.h"
#include "Macros.h"

EditorCamera::EditorCamera(Object *parent, int id) : Camera(parent, id) {}

EditorCamera::~EditorCamera() = default;

void EditorCamera::OnCreate() {
    Camera::OnCreate();

    if (!editorCamera) {
        editorCamera = parent;
        activeCamera = parent;
    }
}

void EditorCamera::Update() {
    if (!enabled) return;

    Camera::Update();

    if (activeCamera != parent) return;

    InputManager* inputManager = InputManager::GetInstance();
    Transform* transform = parent->transform;

    Application* application = Application::GetInstance();
    Viewport* viewport = &Application::viewports[0];
    double cursorX, cursorY;
    glfwGetCursorPos(application->window, &cursorX, &cursorY);

    if (inputManager->IsKeyDown(Key::MOUSE_RIGHT_BUTTON) &&
    cursorX >= viewport->position.x && cursorX <= viewport->position.x + viewport->resolution.x &&
    cursorY <= Application::resolution. y - viewport->position.y &&
    cursorY >= Application::resolution. y - (viewport->position.y + viewport->resolution.y)) {
        cursorPreviousX = cursorX;
        cursorPreviousY = cursorY;
    }

    if (inputManager->IsKeyPressed(Key::MOUSE_RIGHT_BUTTON) && cursorPreviousX > 0 && cursorPreviousY > 0) {
        glfwSetInputMode(application->window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        double mouseX = cursorX - cursorPreviousX;
        double mouseY = cursorY - cursorPreviousY;

        float yaw;
        float pitch;

        if (mouseX != 0.0) {
            int direction = (int)mouseX/abs((int)mouseX);
            yaw = (float)(direction);
        }
        else {
            yaw = 0.0f;
        }
        if (mouseY != 0.0) {
            int direction = (int)mouseY/abs((int)mouseY);
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

    //TODO: move it somewhere else, add checker if rendering camera == nullptr
    if (inputManager->IsKeyPressed(Key::KEY_LEFT_CONTROL) && inputManager->IsKeyDown(Key::KEY_KP_5) ||
        inputManager->IsKeyDown(Key::KEY_LEFT_CONTROL) && inputManager->IsKeyPressed(Key::KEY_KP_5)) {
        renderingCamera->transform->SetLocalPosition(parent->transform->GetGlobalPosition());
        renderingCamera->transform->SetLocalRotation(parent->transform->GetGlobalRotation());
    }
}
