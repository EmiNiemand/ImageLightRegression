#ifndef IMAGELIGHTREGRESSION_RENDERINGMANAGER_H
#define IMAGELIGHTREGRESSION_RENDERINGMANAGER_H

#include "ApplicationTypes.h"

#include <vector>

class Shader;
class Renderer;
class ShadowRenderer;
class ObjectRenderer;
class SkyboxRenderer;
class UIRenderer;
class PostProcessRenderer;

class RenderingManager {
public:
    std::vector<unsigned char> currentlyRenderedImage;

    ShadowRenderer* shadowRenderer = nullptr;
    ObjectRenderer* objectRenderer = nullptr;
    SkyboxRenderer* skyboxRenderer = nullptr;
    UIRenderer* uiRenderer = nullptr;
    PostProcessRenderer* postProcessRenderer = nullptr;

private:
    inline static RenderingManager* renderingManager;
    Shader* selectedObjectShader = nullptr;
    Shader* imageDifferenceShader = nullptr;

    std::vector<Renderer*> drawBuffer{};
public:
    RenderingManager(RenderingManager &other) = delete;
    void operator=(const RenderingManager&) = delete;
    virtual ~RenderingManager();

    static RenderingManager* GetInstance();

    void Startup();
    void Shutdown();

    void Draw(Shader* inShader);
    void DrawFrame();

    void AddToDrawBuffer(Renderer* renderer);
    [[nodiscard]] const std::vector<Renderer*>& GetDrawBuffer() const;

    void UpdateProjection() const;
    void UpdateView() const;

    void OnWindowResize() const;

    void DrawOtherViewports();

private:
    explicit RenderingManager();

    void DrawFrameToTexture(unsigned int fbo);
    void DrawSelectedObjectTexture();
    void DrawPostProcesses();

    void ClearBuffer();
};


#endif //IMAGELIGHTREGRESSION_RENDERINGMANAGER_H
