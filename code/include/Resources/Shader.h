#ifndef IMAGELIGHTREGRESSION_SHADER_H
#define IMAGELIGHTREGRESSION_SHADER_H

#include "Core/Resource.h"

#include "glad/glad.h"
#include "glm/matrix.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include <string>

class Shader : public Resource {
private:
    GLuint shaderID = -1;
public:
    explicit Shader(const std::string &inPath);
    ~Shader() override;

    void Load() override;

    [[nodiscard]] GLuint GetShader() const;

    void Activate() const;
    void Delete() const;

    void SetBool(const std::string &name, bool value) const;
    void SetInt(const std::string &name, int value) const;
    void SetFloat(const std::string &name, float value) const;
    void SetVec2(const std::string &name, const glm::vec2 &value) const;
    void SetIVec2(const std::string &name, const glm::ivec2 &value) const;
    void SetVec2(const std::string &name, float x, float y) const;
    void SetIVec2(const std::string &name, int x, int y) const;
    void SetVec3(const std::string &name, const glm::vec3 &value) const;
    void SetVec3(const std::string &name, float x, float y, float z) const;
    void SetVec4(const std::string &name, const glm::vec4 &value) const;
    void SetVec4(const std::string &name, float x, float y, float z, float w);
    void SetMat2(const std::string &name, const glm::mat2 &mat) const;
    void SetMat3(const std::string &name, const glm::mat3 &mat) const;
    void SetMat4(const std::string &name, const glm::mat4 &mat) const;

private:
    static void LoadShader(std::string& shaderPath, std::string& shaderCodeOut);
};


#endif //IMAGELIGHTREGRESSION_SHADER_H
