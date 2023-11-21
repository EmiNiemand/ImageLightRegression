#ifndef IMAGELIGHTREGRESSION_APPLICATION_H
#define IMAGELIGHTREGRESSION_APPLICATION_H

#include "ApplicationTypes.h"
#include "Structures.h"

#include <GLFW/glfw3.h> // Include glfw3.h after our OpenGL definitions
#include <unordered_map>

class Object;
class Component;

class Application {
public:
    std::unordered_map<int, Object*> objects;
    std::unordered_map<int, Component*> components;

    inline static glm::ivec2 resolution = glm::ivec2(1600, 900);
    inline static Viewport viewports[4];

    float frameTime = 0;

    bool isStarted = false;

private:
    bool shouldRun = false;
    
    inline static Application* application;

    std::vector<int> destroyObjectBuffer{};
    std::vector<int> destroyComponentBuffer{};

public:
    GLFWwindow* window = nullptr;
    Application(Application &other) = delete;
    virtual ~Application();
    void operator=(const Application&) = delete;

    static Application* GetInstance();

    void Startup();
    void Run();
    void Shutdown();

    void AddObjectToDestroyBuffer(int objectID);
    void AddComponentToDestroyBuffer(int componentID);

    void DestroyQueuedComponents();
    void DestroyQueuedObjects();

private:
    Application();
    void CreateApplicationWindow();
    static void glfwErrorCallback(int error, const char* description);
    static void glfwFramebufferSizeCallback(GLFWwindow* window, int width, int height);
};


#endif //IMAGELIGHTREGRESSION_APPLICATION_H
