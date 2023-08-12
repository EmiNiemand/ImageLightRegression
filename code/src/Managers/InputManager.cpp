#include "Managers/InputManager.h"
#include "Application.h"

InputManager::InputManager() = default;

InputManager::~InputManager() = default;

void InputManager::Startup() {
    glfwSetKeyCallback(Application::GetInstance()->window, InputManager::KeyActionCallback);
    glfwSetMouseButtonCallback(Application::GetInstance()->window, InputManager::MouseActionCallback);

    keysDown.reserve(10);
    keysUp.reserve(10);
    keysPressed.reserve(10);
}

void InputManager::Shutdown() {
    glfwSetKeyCallback(Application::GetInstance()->window, nullptr);
    glfwSetMouseButtonCallback(Application::GetInstance()->window, nullptr);
    delete inputManager;
}

InputManager* InputManager::GetInstance() {
    if (inputManager == nullptr) {
        inputManager = new InputManager();
    }
    return inputManager;
}

void InputManager::ManageInput() {
    keysDown.clear();
    keysPressed.clear();
    keysUp.clear();
    keysDown = keysDownBuffer;
    keysPressed = keysPressedBuffer;
    keysUp = keysUpBuffer;
    keysDownBuffer.clear();
    keysUpBuffer.clear();
}

bool InputManager::IsKeyDown(Key key) {
    if (!keysDown.empty()) {
        for (int i = 0; i < keysDown.size(); ++i) {
            if (keysDown[i] == key) {
                return true;
            }
        }
    }
    return false;
}

bool InputManager::IsKeyPressed(Key key) {
    if (!keysPressed.empty()) {
        for (int i = 0; i < keysPressed.size(); ++i) {
            if (keysPressed[i] == key) {
                return true;
            }
        }
    }
    return false;
}

bool InputManager::IsKeyReleased(Key key) {
    if (!keysUp.empty()) {
        for (int i = 0; i < keysUp.size(); ++i) {
            if (keysUp[i] == key) {
                return true;
            }
        }
    }
    return false;
}

void InputManager::KeyActionCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    // Key down - first frame when clicked
    // Key pressed - key down and all frames until released
    if (action == GLFW_PRESS) {
        keysDownBuffer.push_back(static_cast<Key>(key));
        keysPressedBuffer.push_back(static_cast<Key>(key));
    }
    // Key up - released
    if (action == GLFW_RELEASE) {
        if (!keysDownBuffer.empty()) {
            auto iterator = std::find(keysDownBuffer.begin(), keysDownBuffer.end(), static_cast<Key>(key));
            if(iterator != keysDownBuffer.end()) keysDownBuffer.erase(iterator);
        }
        if (!keysPressedBuffer.empty()) {
            auto iterator = std::find(keysPressedBuffer.begin(), keysPressedBuffer.end(), static_cast<Key>(key));
            if(iterator != keysPressedBuffer.end()) keysPressedBuffer.erase(iterator);
        }
        keysUpBuffer.push_back(static_cast<Key>(key));
    }
}

void InputManager::MouseActionCallback(GLFWwindow *window, int button, int action, int mods) {
    // Key down - first frame when clicked
    // Key pressed - key down and all frames until released
    if (action == GLFW_PRESS) {
        keysDownBuffer.push_back(static_cast<Key>(button));
        keysPressedBuffer.push_back(static_cast<Key>(button));
    }
    // Key up - released
    if (action == GLFW_RELEASE) {
        if (!keysDownBuffer.empty()) {
            auto iterator = std::find(keysDownBuffer.begin(), keysDownBuffer.end(), static_cast<Key>(button));
            if(iterator != keysDownBuffer.end()) keysDownBuffer.erase(iterator);
        }
        if (!keysPressedBuffer.empty()) {
            auto iterator = std::find(keysPressedBuffer.begin(), keysPressedBuffer.end(), static_cast<Key>(button));
            if(iterator != keysPressedBuffer.end()) keysPressedBuffer.erase(iterator);
        }
        keysUpBuffer.push_back(static_cast<Key>(button));
    }
}
