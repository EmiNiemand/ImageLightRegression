#ifndef IMAGELIGHTREGRESSION_TRANSFORM_H
#define IMAGELIGHTREGRESSION_TRANSFORM_H

#include "Component.h"

#include "glm/matrix.hpp"
#include "glm/gtc/matrix_transform.hpp"

class Transform : public Component {
private:
    //Global space information concatenate in matrix
    glm::mat4 mModelMatrix = glm::mat4(1.0f);

    //Local space information
    glm::vec3 position = {0.0f, 0.0f, 0.0f };
    glm::vec3 rotation = {0.0f, 0.0f, 0.0f }; //In degrees
    glm::vec3 scale = {1.0f, 1.0f, 1.0f };

    glm::vec3 globalRotation = {0, 0, 0};

    bool dirtyFlag = true;

public:
    Transform(Object *parent, int id);
    ~Transform() override;

    void ComputeModelMatrix();
    void ComputeModelMatrix(const glm::mat4& parentGlobalModelMatrix);

    void SetLocalPosition(const glm::vec3& newPosition);
    void SetLocalRotation(const glm::vec3& newRotation);
    void SetLocalScale(const glm::vec3& newScale);
    void SetDirtyFlag();

    [[nodiscard]] glm::vec3 GetGlobalPosition() const;
    [[nodiscard]] glm::vec3 GetLocalPosition() const;
    /// Return vec3 of rotation values in degrees
    [[nodiscard]] glm::vec3 GetLocalRotation() const;
    [[nodiscard]] glm::vec3 GetGlobalRotation() const;
    [[nodiscard]] glm::vec3 GetLocalScale() const;
    [[nodiscard]] glm::vec3 GetGlobalScale() const;
    [[nodiscard]] glm::mat4 GetModelMatrix() const;
    [[nodiscard]] glm::vec3 GetRight() const;
    [[nodiscard]] glm::vec3 GetUp() const;
    [[nodiscard]] glm::vec3 GetBackward() const;
    [[nodiscard]] glm::vec3 GetForward() const;
    [[nodiscard]] glm::mat4 GetLocalModelMatrix() const;
    [[nodiscard]] bool GetDirtyFlag();

    void Save(nlohmann::json &json) override;
    void Load(nlohmann::json &json) override;

private:
    void CalculateGlobalRotation();
};


#endif //IMAGELIGHTREGRESSION_TRANSFORM_H
