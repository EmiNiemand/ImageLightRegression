#ifndef IMAGELIGHTREGRESSION_DIRECTIONALLIGHT_H
#define IMAGELIGHTREGRESSION_DIRECTIONALLIGHT_H

#include "Components/Component.h"

#include "glm/matrix.hpp"
#include "glm/gtc/matrix_transform.hpp"

class DirectionalLight : public Component {
private:
    glm::vec3 ambient = {0.4f, 0.4f, 0.4f};
    glm::vec3 diffuse = {0.69f, 0.69f, 0.69f};
    glm::vec3 specular = {0.9f, 0.9f, 0.9f};
    glm::vec3 color = {1.0f, 1.0f, 1.0f};

public:
    DirectionalLight(Object* parent, int id);
    ~DirectionalLight() override;

    void OnCreate() override;
    void OnDestroy() override;

    void OnUpdate() override;

    [[nodiscard]] const glm::vec3 &GetAmbient() const;
    [[nodiscard]] const glm::vec3 &GetDiffuse() const;
    [[nodiscard]] const glm::vec3 &GetSpecular() const;
    [[nodiscard]] const glm::vec3 &GetColor() const;

    void SetAmbient(const glm::vec3 &inAmbient);
    void SetDiffuse(const glm::vec3 &inDiffuse);
    void SetSpecular(const glm::vec3 &inSpecular);
    void SetColor(const glm::vec3 &inColor);
};


#endif //IMAGELIGHTREGRESSION_DIRECTIONALLIGHT_H
