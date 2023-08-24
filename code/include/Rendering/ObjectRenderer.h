#ifndef IMAGELIGHTREGRESSION_OBJECTRENDERER_H
#define IMAGELIGHTREGRESSION_OBJECTRENDERER_H

#include "ApplicationTypes.h"

#define NUMBER_OF_LIGHTS 4

class Shader;
class PointLight;
class DirectionalLight;
class SpotLight;

class ObjectRenderer {
public:
    // pair of light number and ptr to light
    PointLight* pointLights[NUMBER_OF_LIGHTS]{};
    DirectionalLight* directionalLights[NUMBER_OF_LIGHTS]{};
    SpotLight* spotLights[NUMBER_OF_LIGHTS]{};

    Shader* shader = nullptr;

    unsigned int fbo;
    unsigned int screenTexture, selectedObjectTexture;

private:
    unsigned int rbo;
    unsigned int attachment[2];

public:
    ObjectRenderer();
    virtual ~ObjectRenderer();

    void PrepareBuffers();

    void UpdateLight(int componentId);
    void RemoveLight(int componentId);

private:
    void UpdatePointLight(int id, Shader* lightShader);
    void UpdateDirectionalLight(int id, Shader* lightShader);
    void UpdateSpotLight(int id, Shader* lightShader);
    void RemovePointLight(int id, Shader* lightShader);
    void RemoveDirectionalLight(int id, Shader* lightShader);
    void RemoveSpotLight(int id, Shader* lightShader);
};


#endif //IMAGELIGHTREGRESSION_OBJECTRENDERER_H
