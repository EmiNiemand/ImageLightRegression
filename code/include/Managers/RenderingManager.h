#ifndef IMAGELIGHTREGRESSION_RENDERINGMANAGER_H
#define IMAGELIGHTREGRESSION_RENDERINGMANAGER_H

#include <map>

class Shader;
class Renderer;
class PointLight;
class DirectionalLight;
class SpotLight;

class RenderingManager {
public:
    // pair of id and ptr to light
    std::map<int, PointLight*> pointLights;
    std::map<int, DirectionalLight*> directionalLights;
    std::map<int, SpotLight*> spotLights;

    Shader* shader;
    //Shader* cubeMapShader;

private:
    inline static RenderingManager* renderingManager;
    unsigned int bufferIterator = 0;
    Renderer* drawBuffer[1000] = {};

public:
    RenderingManager(RenderingManager &other) = delete;
    void operator=(const RenderingManager&) = delete;
    virtual ~RenderingManager();

    static RenderingManager* GetInstance();

    void Free() const;

    void Draw(Shader* inShader);
    void AddToDrawBuffer(Renderer* renderer);

    void UpdateProjection() const;
    void UpdateView() const;

    void UpdateLight(int componentId);
    void RemoveLight(int componentId);

private:
    explicit RenderingManager();

    void UpdatePointLight(int id, Shader* lightShader);
    void UpdateDirectionalLight(int id, Shader* lightShader);
    void UpdateSpotLight(int id, Shader* lightShader);
    void RemovePointLight(int id, Shader* lightShader);
    void RemoveDirectionalLight(int id, Shader* lightShader);
    void RemoveSpotLight(int id, Shader* lightShader);

    void ClearBuffer();
};


#endif //IMAGELIGHTREGRESSION_RENDERINGMANAGER_H
