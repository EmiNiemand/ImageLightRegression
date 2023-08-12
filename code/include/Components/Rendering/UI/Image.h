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

public:
    Image(Object *parent, int id);
    ~Image() override;

    void OnDestroy() override;

    void Draw(Shader* inShader);

    void SetTexture(const std::string& inPath);
};


#endif //IMAGELIGHTREGRESSION_IMAGE_H
