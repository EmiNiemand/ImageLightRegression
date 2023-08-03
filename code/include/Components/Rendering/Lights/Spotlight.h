#ifndef IMAGELIGHTREGRESSION_SPOTLIGHT_H
#define IMAGELIGHTREGRESSION_SPOTLIGHT_H

#include "Components/Component.h"

#include "glm/matrix.hpp"
#include "glm/gtc/matrix_transform.hpp"

class SpotLight : public Component {
private:
    glm::vec3 ambient = {0.0f, 0.0f, 0.0f};
    glm::vec3 diffuse = {1.0f, 1.0f, 1.0f};
    glm::vec3 specular = {1.0f, 1.0f, 1.0f};
    glm::vec3 color = {1.0f, 1.0f, 1.0f};

    float cutOff = glm::cos(glm::radians(12.5f));
    float outerCutOff = glm::cos(glm::radians(15.0f));

    float constant = 1.0f;
    float linear = 0.045f;
    float quadratic = 0.0075f;

public:
    SpotLight(Object* parent, int id);
    ~SpotLight() override;

    void OnCreate() override;
    void OnDestroy() override;

    void OnUpdate() override;

    float GetCutOff() const;
    float GetOuterCutOff() const;
    float GetConstant() const;
    float GetLinear() const;
    float GetQuadratic() const;
    const glm::vec3 &GetAmbient() const;
    const glm::vec3 &GetDiffuse() const;
    const glm::vec3 &GetSpecular() const;
    const glm::vec3 &GetColor() const;


    void SetCutOff(float inCutOff);
    void SetOuterCutOff(float inOuterCutOff);
    void SetConstant(float inConstant);
    void SetLinear(float inLinear);
    void SetQuadratic(float inQuadratic);
    void SetAmbient(const glm::vec3 &inAmbient);
    void SetDiffuse(const glm::vec3 &inDiffuse);
    void SetSpecular(const glm::vec3 &inSpecular);
    void SetColor(const glm::vec3 &inColor);
};


#endif //IMAGELIGHTREGRESSION_SPOTLIGHT_H
