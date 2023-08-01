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

class Application {
public:
    float frameTime = 0;

private:
    std::pair<int, int> resolution = std::make_pair(800, 600);
    inline static Application* application;
    bool shouldRun = false;

public:
    GLFWwindow* window = nullptr;
    Application(Application &other) = delete;
    virtual ~Application();
    void operator=(const Application&) = delete;

    static Application* GetInstance();

    void StartUp();
    void Run();
    void ShutDown();

private:
    Application();
    void CreateApplicationWindow();
    static void glfwErrorCallback(int error, const char* description);
};


#endif //IMAGELIGHTREGRESSION_APPLICATION_H
