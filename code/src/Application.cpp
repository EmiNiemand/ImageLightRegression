#include "Application.h"
#include "Managers/ResourceManager.h"
#include "Managers/InputManager.h"
#include "Core/Object.h"
#include "Editor/Components/EditorCamera.h"
#include "Components/Transform.h"
#include "Components/Rendering/Renderer.h"
#include "Managers/RenderingManager.h"
#include "Components/Rendering/Lights/PointLight.h"
#include "Components/Rendering/Skybox.h"
#include "Components/Rendering/UI/Image.h"

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

    Skybox::InitializeBuffers();
    Image::InitializeBuffers();

    scene = Object::Instantiate("Scene", nullptr);

    // World viewport - tool
    viewports[0] = {glm::ivec2(resolution.x / 16 * 3, resolution.y / 9 * 2), glm::ivec2(resolution.x / 16 * 10, resolution.y / 9 * 6)};
    // Loaded image viewport
    viewports[1] = {glm::ivec2(0, resolution.y / 9 * 2), glm::ivec2(resolution.x / 16 * 3, resolution.y / 9 * 2)};
    // Rendered image viewport
    viewports[2] = {glm::ivec2(resolution.x / 64 * 35, resolution.y / 18 * 11), glm::ivec2(resolution.x / 16 * 4, resolution.y / 9 * 2)};
    // Calculated difference image viewport
    viewports[3] = {glm::ivec2(resolution.x / 64 * 35, resolution.y / 18 * 5), glm::ivec2(resolution.x / 16 * 4, resolution.y / 9 * 2)};

    Object* mainCamera = Object::Instantiate("Main Camera", scene);
    mainCamera->AddComponent<EditorCamera>();
    Camera::SetActiveCamera(mainCamera);
    mainCamera->transform->SetLocalPosition({0, 0, 10});

    Object* skybox = Object::Instantiate("Skybox", scene);
    skybox->AddComponent<Skybox>();
    Skybox::SetActiveSkybox(skybox);

    Object* loadedObject = Object::Instantiate("Something", scene);
    loadedObject->AddComponent<Renderer>()->LoadModel("resources/models/Cube/Cube.obj");
    loadedObject->GetComponentByClass<Renderer>()->material = {glm::vec3(1.0f, 1.0f, 1.0f), 32.0f, 0.0f, 0.0f};

    image1 = Object::Instantiate("Image1", scene);
    image1->AddComponent<Image>();

    image2 = Object::Instantiate("Image2", scene);
    image2->AddComponent<Image>();

    image3 = Object::Instantiate("Image3", scene);
    image3->AddComponent<Image>();

    Object* pointLight = Object::Instantiate("Point Light", scene);
    pointLight->AddComponent<PointLight>();
    pointLight->transform->SetLocalPosition({10, 1, 0});

    //Object::Destroy(loadedObject);
}

void Application::Run() {
    while(!shouldRun) {
        frameTime = (float)glfwGetTime();

        InputManager::GetInstance()->ManageInput();

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        for (int i = 0; i < destroyComponentBufferIterator; ++i) {
            int componentID = destroyComponentBuffer[i];
            components[componentID]->OnDestroy();
            delete components[componentID];
            components.erase(componentID);
        }
        destroyComponentBufferIterator = 0;

        for (int i = 0; i < destroyObjectBufferIterator; ++i) {
            int objectID = destroyObjectBuffer[i];
            objects[objectID]->parent->children.erase(objectID);
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

        for (const auto& component: components) {
            if (component.second->enabled) {
                component.second->Update();
            }
        }

        glViewport(viewports[0].position.x, viewports[0].position.y, viewports[0].resolution.x, viewports[0].resolution.y);
        RenderingManager::GetInstance()->Draw(RenderingManager::GetInstance()->shader);
        Skybox::Draw(RenderingManager::GetInstance()->cubeMapShader);

        glViewport(viewports[1].position.x, viewports[1].position.y, viewports[1].resolution.x, viewports[1].resolution.y);
        // show
        image1->GetComponentByClass<Image>()->Draw(RenderingManager::GetInstance()->imageShader);

        if (isStarted) {
            glViewport(viewports[2].position.x, viewports[2].position.y, viewports[2].resolution.x, viewports[2].resolution.y);
            // show
            image2->GetComponentByClass<Image>()->Draw(RenderingManager::GetInstance()->imageShader);
            glViewport(viewports[3].position.x, viewports[3].position.y, viewports[3].resolution.x, viewports[3].resolution.y);
            // show
            image3->GetComponentByClass<Image>()->Draw(RenderingManager::GetInstance()->imageShader);
        }


        glfwSwapBuffers(window);
        glfwPollEvents();

        RenderingManager::GetInstance()->ClearBuffer();

        shouldRun = glfwWindowShouldClose(window);
    }
}

void Application::ShutDown() {
    Skybox::DeleteBuffers();
    Image::DeleteBuffers();
    RenderingManager::GetInstance()->Shutdown();
    glfwDestroyWindow(window);
    glfwTerminate();

    delete application;
}

void Application::AddObjectToDestroyBuffer(int objectID) {
    destroyObjectBuffer[destroyObjectBufferIterator] = objectID;
    ++destroyObjectBufferIterator;
}

void Application::AddComponentToDestroyBuffer(int componentID) {
    destroyComponentBuffer[destroyComponentBufferIterator] = componentID;
    ++destroyComponentBufferIterator;
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
    window = glfwCreateWindow(resolution.x, resolution.y, "ILR", NULL, NULL);
    if (window == nullptr)
        throw;
    glfwMakeContextCurrent(window);
    // Enable vsync
    glfwSwapInterval(false);

    // Center window on the screen
    const GLFWvidmode * mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
    int monitorWidth = mode->width;
    int monitorHeight = mode->height;
    glfwSetWindowPos(window, monitorWidth / 2 - resolution.x / 2, monitorHeight / 2 - resolution.y / 2);
    glfwSetFramebufferSizeCallback(window, glfwFramebufferSizeCallback);

    // Enable cursor - change last parameter to disable it
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    bool err = !gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    if (err)
    {
        spdlog::error("Failed to initialize OpenGL loader!");
        throw;
    }
    spdlog::info("Successfully initialized OpenGL loader!");

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    stbi_set_flip_vertically_on_load(true);
}

void Application::glfwErrorCallback(int error, const char *description) {
    spdlog::error("Glfw Error" + std::to_string(error) + ": " + description);
}

void Application::glfwFramebufferSizeCallback(GLFWwindow *window, int width, int height) {
    resolution.x = width;
    resolution.y = height;

    // World viewport - tool
    viewports[0] = {glm::ivec2(resolution.x / 16 * 3, resolution.y / 9 * 2), glm::ivec2(resolution.x / 16 * 10, resolution.y / 9 * 6)};
    // Loaded image viewport
    viewports[1] = {glm::ivec2(0, resolution.y / 9 * 2), glm::ivec2(resolution.x / 16 * 3, resolution.y / 9 * 2)};
    // Rendered image viewport
    viewports[2] = {glm::ivec2(resolution.x / 64 * 35, resolution.y / 18 * 11), glm::ivec2(resolution.x / 16 * 4, resolution.y / 9 * 2)};
    // Calculated difference image viewport
    viewports[3] = {glm::ivec2(resolution.x / 64 * 35, resolution.y / 18 * 5), glm::ivec2(resolution.x / 16 * 4, resolution.y / 9 * 2)};

}
