#ifndef IMAGELIGHTREGRESSION_RENDERINGMANAGER_H
#define IMAGELIGHTREGRESSION_RENDERINGMANAGER_H

#include "ApplicationTypes.h"

#include <vector>

class Shader;
class Renderer;
class ShadowRenderer;
class ObjectRenderer;
class SkyboxRenderer;

class RenderingManager {
public:
    ShadowRenderer* shadowRenderer = nullptr;
    ObjectRenderer* objectRenderer = nullptr;
    SkyboxRenderer* skyboxRenderer = nullptr;

private:
    inline static RenderingManager* renderingManager;

    std::vector<Renderer*> drawBuffer{};
public:
    RenderingManager(RenderingManager &other) = delete;
    void operator=(const RenderingManager&) = delete;
    virtual ~RenderingManager();

    static RenderingManager* GetInstance();

    void Startup();
    void Shutdown();

    void Draw(Shader* inShader);
    void ClearBuffer();

    void AddToDrawBuffer(Renderer* renderer);
    [[nodiscard]] const std::vector<Renderer*>& GetDrawBuffer() const;

    void UpdateProjection() const;
    void UpdateView() const;

private:
    explicit RenderingManager();
};


#endif //IMAGELIGHTREGRESSION_RENDERINGMANAGER_H
