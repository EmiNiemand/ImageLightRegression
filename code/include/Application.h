#ifndef IMAGELIGHTREGRESSION_APPLICATION_H
#define IMAGELIGHTREGRESSION_APPLICATION_H

#pragma region "Library includes"
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include <cstdio>

#define IMGUI_IMPL_OPENGL_LOADER_GLAD

#if defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
#include <glad/glad.h>  // Initialize with gladLoadGL()
#else
#include IMGUI_IMPL_OPENGL_LOADER_CUSTOM
#endif

#include <GLFW/glfw3.h> // Include glfw3.h after our OpenGL definitions
#include <spdlog/spdlog.h>
#include <stb_image.h>
#pragma endregion

#include "ApplicationTypes.h"
#include "Structures.h"

class Object;
class Component;

class Application {
public:
    std::unordered_map<int, Object*> objects;
    std::unordered_map<int, Component*> components;

    inline static glm::ivec2 resolution = glm::ivec2(1920, 1000);
    inline static Viewport viewports[4];

    Object* scene = nullptr;
    float frameTime = 0;

private:
    inline static Application* application;

    unsigned int destroyObjectBufferIterator = 0;
    int destroyObjectBuffer[200]{};
    unsigned int destroyComponentBufferIterator = 0;
    int destroyComponentBuffer[200]{};

    bool shouldRun = false;
    bool isStarted = false;

public:
    GLFWwindow* window = nullptr;
    Application(Application &other) = delete;
    virtual ~Application();
    void operator=(const Application&) = delete;

    static Application* GetInstance();

    void StartUp();
    void Run();
    void ShutDown();

    void AddObjectToDestroyBuffer(int objectID);
    void AddComponentToDestroyBuffer(int componentID);

private:
    Application();
    void CreateApplicationWindow();
    static void glfwErrorCallback(int error, const char* description);
    static void glfwFramebufferSizeCallback(GLFWwindow* window, int width, int height);
};


#endif //IMAGELIGHTREGRESSION_APPLICATION_H
