#ifndef IMAGELIGHTREGRESSION_SHADOWRENDERER_H
#define IMAGELIGHTREGRESSION_SHADOWRENDERER_H

#include "glm/glm.hpp"

class Shader;

class ShadowRenderer {
public:
    Shader* shadowShader = nullptr;

    unsigned int depthMapFBO = 0;
    unsigned int depthMap = 0;

    int shadowResolution = 4096;

    glm::mat4 lightSpaceMatrix;

public:
    ShadowRenderer();
    virtual ~ShadowRenderer();

    void PrepareShadowMap();
};


#endif //IMAGELIGHTREGRESSION_SHADOWRENDERER_H
