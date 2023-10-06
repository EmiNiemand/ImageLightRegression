#ifndef IMAGELIGHTREGRESSION_SHADOWRENDERER_H
#define IMAGELIGHTREGRESSION_SHADOWRENDERER_H

#include "glm/glm.hpp"

class Shader;

class ShadowRenderer {
public:
    Shader* dnslShadowShader = nullptr;
    Shader* plShadowShader = nullptr;

    unsigned int depthMapFBOs[12];
    unsigned int depthMaps[12];

    glm::mat4 directionalLightSpaceMatrices[4];
    glm::mat4 spotLightSpaceMatrices[4];

public:
    ShadowRenderer();
    virtual ~ShadowRenderer();

    void PrepareShadowMap();
};


#endif //IMAGELIGHTREGRESSION_SHADOWRENDERER_H
