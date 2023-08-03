#ifndef IMAGELIGHTREGRESSION_RENDERER_H
#define IMAGELIGHTREGRESSION_RENDERER_H

#include "Components/Component.h"

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include <string>

class Shader;
class Model;

struct Material {
    glm::vec3 color;
    float shininess;
    // Values: <0, 1>
    float reflection;
    // Values: <0, 1>
    float refraction;
};


class Renderer : public Component {
public:
    Material material = {{1.0f, 1.0f, 1.0f}, 32.0f, 0, 0};
    glm::vec2 textScale = glm::vec2(1.0f, 1.0f);

    Model* model = nullptr;
    bool drawShadows = true;

public:
    Renderer(Object *parent, int id);
    ~Renderer() override;

    void Update() override;
    void Draw(Shader* shader);

    void LoadModel(const std::string& path);

private:
    void AddToDraw();
};


#endif //IMAGELIGHTREGRESSION_RENDERER_H
