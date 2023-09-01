#ifndef IMAGELIGHTREGRESSION_POINTLIGHT_H
#define IMAGELIGHTREGRESSION_POINTLIGHT_H

#include "Components/Component.h"

#include "glm/matrix.hpp"
#include "glm/gtc/matrix_transform.hpp"

class PointLight : public Component {
private:
    glm::vec3 ambient = {1.0f, 1.0f, 1.0f};
    glm::vec3 diffuse = {0.69f, 0.69f, 0.69f};
    glm::vec3 specular = {1.0f, 1.0f, 1.0f};
    glm::vec3 color = {1.0f, 1.0f, 1.0f};

    float constant = 1.0f;
    float linear = 0.007f;
    float quadratic = 0.0002f;
public:
    PointLight(Object* parent, int id);
    ~PointLight() override;

    void OnCreate() override;
    void OnDestroy() override;

    void OnUpdate() override;

    [[nodiscard]] float GetConstant() const;
    [[nodiscard]] float GetLinear() const;
    [[nodiscard]] float GetQuadratic() const;
    [[nodiscard]] const glm::vec3& GetAmbient() const;
    [[nodiscard]] const glm::vec3& GetDiffuse() const;
    [[nodiscard]] const glm::vec3& GetSpecular() const;
    [[nodiscard]] const glm::vec3& GetColor() const;

    void SetConstant(float inConstant);
    void SetLinear(float inLinear);
    void SetQuadratic(float inQuadratic);
    void SetAmbient(const glm::vec3& inAmbient);
    void SetDiffuse(const glm::vec3& inDiffuse);
    void SetSpecular(const glm::vec3& inSpecular);
    void SetColor(const glm::vec3& inColor);

    void Save(nlohmann::json &json) override;
    void Load(nlohmann::json &json) override;
};


#endif //IMAGELIGHTREGRESSION_POINTLIGHT_H
