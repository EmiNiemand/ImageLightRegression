#include "Application.h"
#include "Managers/ResourceManager.h"
#include "Core/Object.h"
#include "Components/Component.h"

Application::Application() = default;

Application::~Application() = default;

Application *Application::GetInstance() {
    if (application == nullptr) {
        application = new Application();
    }
    return application;
}

void Application::StartUp() {
    CreateApplicationWindow();

    scene = Object::Instantiate("Scene", nullptr);
}

void Application::Run() {
    while(!shouldRun) {
        frameTime = (float)glfwGetTime();
        glfwPollEvents();
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

        for (int i = 0; i < destroyComponentBufferIterator; ++i) {
            int componentID = destroyComponentBuffer[i];
            components[componentID]->OnDestroy();
            delete components[componentID];
            components.erase(componentID);
        }
        destroyComponentBufferIterator = 0;

        for (int i = 0; i < destroyObjectBufferIterator; ++i) {
            int objectID = destroyObjectBuffer[i];
            objects[objectID]->Destroy();
            delete objects[objectID];
            objects.erase(objectID);
        }
        destroyObjectBufferIterator = 0;

        scene->UpdateSelfAndChildren();

        for (const auto& component: components) {
            if (component.second->callOnAwake) {
                component.second->Awake();
                component.second->parent->UpdateSelfAndChildren();
            }
        }

        for (const auto& component: components) {
            if (component.second->callOnStart && component.second->enabled) {
                component.second->Start();
                component.second->parent->UpdateSelfAndChildren();
            }
        }


        glfwSwapBuffers(window);
        shouldRun = glfwWindowShouldClose(window);
    }
}

void Application::ShutDown() {
    glfwDestroyWindow(window);
    glfwTerminate();

    delete application;
}

void Application::CreateApplicationWindow() {
    // Setup window
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit())
        throw;

    // Decide GL+GLSL versions
    // GL 4.3 + GLSL 430
    const char* glsl_version = "#version 430";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
    glfwWindowHint(GLFW_SAMPLES, 4);
    // Create window with graphics context
    window = glfwCreateWindow(resolution.first, resolution.second, "ILR", NULL, NULL);
    if (window == nullptr)
        throw;
    glfwMakeContextCurrent(window);
    // Enable vsync
    glfwSwapInterval(false);

    // Center window on the screen
    const GLFWvidmode * mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
    int monitorWidth = mode->width;
    int monitorHeight = mode->height;
    glfwSetWindowPos(window, monitorWidth / 2 - resolution.first / 2, monitorHeight / 2 - resolution.second / 2);

    // Enable cursor - change last parameter to disable it
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    bool err = !gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    if (err)
    {
        spdlog::error("Failed to initialize OpenGL loader!");
        throw;
    }
    spdlog::info("Successfully initialized OpenGL loader!");


    // DebugManager::GetInstance()->Initialize(window, glsl_version);

    glEnable(GL_MULTISAMPLE);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    stbi_set_flip_vertically_on_load(true);
}

void Application::glfwErrorCallback(int error, const char *description) {
    spdlog::error("Glfw Error" + std::to_string(error) + ": " + description);
}

void Application::AddObjectToDestroyBuffer(int objectID) {
    destroyObjectBuffer[destroyObjectBufferIterator] = objectID;
    ++destroyObjectBufferIterator;
}

void Application::AddComponentToDestroyBuffer(int componentID) {
    destroyComponentBuffer[destroyComponentBufferIterator] = componentID;
    ++destroyComponentBufferIterator;
}
