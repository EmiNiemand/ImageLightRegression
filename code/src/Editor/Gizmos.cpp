#include "Editor/Gizmos.h"
#include "Managers/ResourceManager.h"
#include "Managers/EditorManager.h"
#include "Managers/InputManager.h"
#include "Managers/SceneManager.h"
#include "Resources/Shader.h"
#include "Core/Object.h"
#include "Components/Transform.h"
#include "Components/Rendering/Camera.h"
#include "Structures.h"
#include "CUM.h"

#define M_PI 3.14159265358979323846

Gizmos::Gizmos() {
    gizmoShader = ResourceManager::LoadResource<Shader>("resources/Resources/ShaderResources/GizmosShader.json");

    hookPoints.reserve(72);

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Point::vertices), &Point::vertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
}

Gizmos::~Gizmos() {
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);

    gizmoShader->Delete();
    ResourceManager::UnloadResource(gizmoShader->GetPath());
}

void Gizmos::Draw() {
    Object* selectedObject = EditorManager::GetInstance()->selectedNode;
    if (!selectedObject) return;
    gizmoShader->Activate();
    gizmoShader->SetMat4("model", selectedObject->transform->GetModelMatrix());
    gizmoShader->SetInt("mode", mode);

    glBindVertexArray(vao);
    glDrawArrays(GL_POINTS, 0, 1);
    glBindVertexArray(0);
}

void Gizmos::Update() {
    Object* selectedObject = EditorManager::GetInstance()->selectedNode;
    if (!selectedObject || selectedObject == SceneManager::GetInstance()->scene) return;

    hookPoints.clear();

    // Calculate hook points
    if (mode != 1) {
        for (int i = 0; i < 3; ++i) {
            glm::vec3 direction = glm::vec3(0.0f);
            direction[i] = 1.0f;
            CalculateLinePoint(selectedObject->transform->GetGlobalPosition(), direction);
        }
    }
    else {
        CalculateCirclePoints(selectedObject->transform->GetGlobalPosition(), glm::vec3(1, 0, 0), glm::vec3(0, 0, 1));
        CalculateCirclePoints(selectedObject->transform->GetGlobalPosition(), glm::vec3(0, 1, 0), glm::vec3(0, 0, 1));
        CalculateCirclePoints(selectedObject->transform->GetGlobalPosition(), glm::vec3(1, 0, 0), glm::vec3(0, 1, 0));
    }

    ManageInput();
}

void Gizmos::ManageInput() {
    Object* selectedObject = EditorManager::GetInstance()->selectedNode;

    // Change gizmo mode
    InputManager* inputManager = InputManager::GetInstance();
    if (inputManager->IsKeyPressed(Key::KEY_LEFT_CONTROL) && inputManager->IsKeyDown(Key::KEY_E)) {
        mode = 0;
    }
    else if (inputManager->IsKeyPressed(Key::KEY_LEFT_CONTROL) && inputManager->IsKeyDown(Key::KEY_R)) {
        mode = 1;
    }
    else if (inputManager->IsKeyPressed(Key::KEY_LEFT_CONTROL) && inputManager->IsKeyDown(Key::KEY_T)) {
        mode = 2;
    }

    Application* application = Application::GetInstance();
    Viewport* viewport = &Application::viewports[0];

    double cursorX, cursorY;
    glfwGetCursorPos(application->window, &cursorX, &cursorY);

    glm::vec2 cursorPosition = glm::vec2(cursorX, cursorY);

    // Check if mouse is in viewport if left mouse was clicked
    if (inputManager->IsKeyDown(Key::MOUSE_LEFT_BUTTON) && CUM::IsInViewport(cursorPosition, viewport)) {
        cursorPreviousX = cursorX;
        cursorPreviousY = cursorY;

        glm::vec2 viewportResolution = glm::vec2(viewport->resolution);
        glm::vec2 viewportPosition = glm::vec2(viewport->position);

        // Calculate window space to viewport space
        cursorPosition.x = 2.0f * (cursorPosition.x - viewportPosition.x) / viewportResolution.x - 1.0f;
        cursorPosition.y = 1.0f - 2.0f * (cursorPosition.y - ((float)Application::resolution.y - (viewportPosition.y +
                                                               viewportResolution.y))) / viewportResolution.y;

        // Look for closest point and select it as hooked point if it's close enough to cursor
        float minDistance = 0.0f;
        float distance;
        for (int8 i = 0; i < (int8)hookPoints.size(); ++i) {
            distance = glm::distance(hookPoints[i], cursorPosition);
            if (hookedPoint == -1 && distance <= 0.05f) {
                hookedPoint = i;
                minDistance = distance;
            }
            else if (hookedPoint >= 0 && distance < minDistance) {
                hookedPoint = i;
            }
        }
    }
    // Manage mouse input if any point was hooked
    if (inputManager->IsKeyPressed(Key::MOUSE_LEFT_BUTTON) && hookedPoint >= 0) {
        glfwSetInputMode(application->window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        float mouseShift;
        int direction = hookedPoint % 3;

        if (hookPoints.size() > 3) {
            direction = hookedPoint / 24;
        }

        if (direction == 0) {
            mouseShift = (float)(cursorX - cursorPreviousX);
        }
        else if (direction == 1) {
            mouseShift = (float)(cursorPreviousY - cursorY);
        }
        else {
            mouseShift = (float)(cursorX - cursorPreviousX + cursorY - cursorPreviousY) / 2;
        }

        glm::vec3 transformValue = glm::vec3(0.0f);
        transformValue[direction] = mouseShift;

        if (mode == 0) {
            selectedObject->transform->SetLocalPosition(selectedObject->transform->GetGlobalPosition() + transformValue);
        }
        else if (mode == 1) {
            selectedObject->transform->SetLocalRotation(selectedObject->transform->GetGlobalRotation() + transformValue);
        }
        else {
            selectedObject->transform->SetLocalScale(selectedObject->transform->GetGlobalScale() + transformValue);
        }

        glfwSetCursorPos(application->window, cursorPreviousX, cursorPreviousY);
    }
    // Reset state after mouse button was released
    else if (inputManager->IsKeyReleased(Key::MOUSE_LEFT_BUTTON)) {
        glfwSetInputMode(Application::GetInstance()->window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        hookedPoint = -1;
    }
}

void Gizmos::CalculateLinePoint(glm::vec3 initialPosition, glm::vec3 direction) {
    Camera* camera = Camera::GetActiveCamera()->GetComponentByClass<Camera>();

    glm::mat4 view = camera->GetViewMatrix();
    glm::mat4 projectionView = camera->GetProjectionMatrix() * view;

    float size = glm::length(glm::vec3(view[3])) / 3;

    direction *= 0.25f * size;

    glm::vec4 objectPosition = glm::vec4(initialPosition, 1.0f);

    objectPosition += glm::vec4(direction, 0.0f);
    objectPosition = projectionView * objectPosition;

    glm::vec2 objectViewportPosition = glm::vec2(objectPosition.x, objectPosition.y) / objectPosition.w;

    hookPoints.emplace_back(objectViewportPosition);
}

void Gizmos::CalculateCirclePoints(glm::vec3 initialPosition, glm::vec3 direction1, glm::vec3 direction2) {
    Camera* camera = Camera::GetActiveCamera()->GetComponentByClass<Camera>();

    glm::mat4 view = camera->GetViewMatrix();
    glm::mat4 projectionView = camera->GetProjectionMatrix() * view;
    // Size scalar to keep gizmos the same size even when camera is moving away
    float size = glm::length(glm::vec3(view[3])) / 3;

    for (int j = 0; j < 24; ++j) {
        glm::vec4 objectPosition = glm::vec4(initialPosition, 1.0f);

        float ang = M_PI * 2 / 24 * j;

        glm::vec3 offset = (direction1 * cos(ang) + direction2 * -sin(ang));
        objectPosition = projectionView * (objectPosition + glm::vec4(offset, 0) * 0.25f * size);

        glm::vec2 objectViewportPosition = glm::vec2(objectPosition.x, objectPosition.y) / objectPosition.w;

        hookPoints.emplace_back(objectViewportPosition);
    }
}

