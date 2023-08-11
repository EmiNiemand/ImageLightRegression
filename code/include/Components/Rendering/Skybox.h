#ifndef IMAGELIGHTREGRESSION_SKYBOX_H
#define IMAGELIGHTREGRESSION_SKYBOX_H

#include "Components/Component.h"
#include "Structures.h"

class Object;
class CubeMap;
class Shader;

class Skybox : public Component {
private:
    inline static Object* activeSkybox = nullptr;
    CubeMap* cubeMap = nullptr;

    inline static unsigned int vao = 0;
    inline static unsigned int vbo = 0;

public:
    Skybox(Object *parent, int id);
    ~Skybox() override;

    void OnDestroy() override;

    static void Draw(Shader* inShader);

    static void SetActiveSkybox(Object* inSkybox);

    /// Call after window creation and opengl init, somewhere at application start up
    static void InitializeBuffers();
    /// Call during application shutdown
    static void DeleteBuffers();
};


#endif //IMAGELIGHTREGRESSION_SKYBOX_H
