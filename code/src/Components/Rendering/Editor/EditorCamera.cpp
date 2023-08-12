#include "Components/Rendering/Editor/EditorCamera.h"
#include "Application.h"
#include "Managers/InputManager.h"
#include "Core/Object.h"
#include "Components/Transform.h"

EditorCamera::EditorCamera(Object *parent, int id) : Camera(parent, id) {}

EditorCamera::~EditorCamera() = default;

void EditorCamera::Update() {
    Component::Update();

    InputManager* inputManager = InputManager::GetInstance();
    Transform* transform = parent->transform;

    if (inputManager->IsKeyPressed(Key::MOUSE_RIGHT_BUTTON)) {
        Application* application = Application::GetInstance();
        glfwSetInputMode(application->window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        double cursorX, cursorY;
        glfwGetCursorPos(application->window, &cursorX, &cursorY);

        if (inputManager->IsKeyDown(Key::MOUSE_RIGHT_BUTTON)) {
            cursorPreviousX = cursorX;
            cursorPreviousY = cursorY;
        }

        double mouseX = cursorX - cursorPreviousX;
        double mouseY = cursorY - cursorPreviousY;

        float yaw;
        float pitch;
        if (abs(mouseX) >= 200.0 && mouseX != 0.0) {
            int direction = (int)mouseX/abs((int)mouseX);
            yaw = (float)(direction);
        }
        else {
            yaw = 0.0f;
        }
        if (abs(mouseY) >= 200.0 && mouseY != 0.0) {
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
    }
    else if (inputManager->IsKeyReleased(Key::MOUSE_RIGHT_BUTTON)) {
        glfwSetInputMode(Application::GetInstance()->window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    }
}
