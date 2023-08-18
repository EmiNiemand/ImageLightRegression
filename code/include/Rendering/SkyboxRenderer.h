#ifndef IMAGELIGHTREGRESSION_SKYBOXRENDERER_H
#define IMAGELIGHTREGRESSION_SKYBOXRENDERER_H

class Shader;
class Object;

class SkyboxRenderer {
public:
    Shader* cubeMapShader;
    unsigned int vao = 0;
    unsigned int vbo = 0;

private:
    Object* activeSkybox = nullptr;

public:
    SkyboxRenderer();
    virtual ~SkyboxRenderer();

    void SetActiveSkybox(Object* inSkybox);
    [[nodiscard]] Object* GetActiveSkybox() const;

    void Draw();
};


#endif //IMAGELIGHTREGRESSION_SKYBOXRENDERER_H
