#ifndef IMAGELIGHTREGRESSION_IMAGE_H
#define IMAGELIGHTREGRESSION_IMAGE_H

#include "Components/Component.h"

#include "glm/glm.hpp"

#include <string>

class Texture;
class Shader;

class Image : public Component {
public:
    glm::vec2 size = {1.0f, 1.0f};

private:
    Texture* texture = nullptr;

    inline static unsigned int vao = 0;
    inline static unsigned int vbo = 0;

public:
    Image(Object *parent, int id);
    ~Image() override;

    void OnDestroy() override;

    void Draw(Shader* inShader);

    void SetTexture(const std::string& inPath);

    //TODO: maybe move it to UIManager with UI shader from rendering manager
    static void InitializeBuffers();
    static void DeleteBuffers();
};


#endif //IMAGELIGHTREGRESSION_IMAGE_H
