#include "Resources/Shader.h"
#include "Macros.h"

#include "nlohmann/json.hpp"

#include <sstream>
#include <fstream>

Shader::Shader(const std::string &inPath) : Resource(inPath) {

}

Shader::~Shader() = default;

void Shader::Load() {
    std::string vertexShaderSource;
    std::string fragmentShaderSource;

#ifdef RELEASE
    std::ifstream input(path);

    nlohmann::json json;
    input >> json;
    json.at("vertex").get_to(vertexShaderSource);
    json.at("fragment").get_to(fragmentShaderSource);
#endif
#ifdef DEBUG
    try {
        std::ifstream input(path);

        nlohmann::json json;
        input >> json;
        json.at("vertex").get_to(vertexShaderSource);
        json.at("fragment").get_to(fragmentShaderSource);
    }
    catch (std::ifstream::failure& e) {
        ILR_ERROR_MSG("Shader file path does not exist" + path);
    }
#endif

    std::string vCode;
    LoadShader(vertexShaderSource, vCode);
    const GLchar* cvCode = vCode.c_str();
    // Generate Shader object
    // Create Vertex Shader Object and get its reference
    GLuint vertexShader;
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    // Attach Vertex Shader source to the Vertex Shader Object
    glShaderSource(vertexShader, 1, &cvCode, nullptr);
    // Compile the Vertex Shader into machine code
    glCompileShader(vertexShader);

    std::string fCode;
    LoadShader(fragmentShaderSource, fCode);
    const GLchar* cfCode = fCode.c_str();
    // Create Fragment Shader Object and get its reference
    GLuint fragmentShader;
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    // Attach Fragment Shader source to the Fragment Shader Object
    glShaderSource(fragmentShader, 1, &cfCode, nullptr);
    // Compile the Vertex Shader into machine code
    glCompileShader(fragmentShader);

    shaderID = glCreateProgram();
    // Attach the Vertex and Fragment Shaders to the Shader Program
    glAttachShader(shaderID, vertexShader);
    glAttachShader(shaderID, fragmentShader);

    // Wrap-up/Link all the shaders together into the Shader Program
    glLinkProgram(shaderID);

    // Delete the now useless Vertex and Fragment Shader objects
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void Shader::LoadShader(std::string& shaderPath, std::string& shaderCode) {
    std::ifstream ShaderFile;

    ShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
#ifdef RELEASE
    ShaderFile.open(shaderPath);
        std::stringstream ShaderStream;

        ShaderStream << ShaderFile.rdbuf();
        ShaderFile.close();

        shaderCode = ShaderStream.str();
#endif

#ifdef DEBUG
    try {
        ShaderFile.open(shaderPath);
        std::stringstream ShaderStream;

        ShaderStream << ShaderFile.rdbuf();
        ShaderFile.close();

        shaderCode = ShaderStream.str();
    }
    catch (std::ifstream::failure& e) {
        ILR_ERROR_MSG("Shader file loading failure" + shaderPath);
    }
#endif
}


GLuint Shader::GetShader() const {
    return shaderID;
}

void Shader::Activate() const {
    glUseProgram(shaderID);
}


void Shader::Delete() const {
    glDeleteProgram(shaderID);
}

#pragma region Utils
void Shader::SetBool(const std::string &name, bool value) const
{
    glUniform1i(glGetUniformLocation(shaderID, name.c_str()), (int)value);
}

void Shader::SetInt(const std::string &name, int value) const
{
    glUniform1i(glGetUniformLocation(shaderID, name.c_str()), value);
}

void Shader::SetFloat(const std::string &name, float value) const
{
    glUniform1f(glGetUniformLocation(shaderID, name.c_str()), value);
}

void Shader::SetVec2(const std::string &name, const glm::vec2 &value) const
{
    glUniform2fv(glGetUniformLocation(shaderID, name.c_str()), 1, &value[0]);
}
void Shader::SetVec2(const std::string &name, float x, float y) const
{
    glUniform2f(glGetUniformLocation(shaderID, name.c_str()), x, y);
}

void Shader::SetVec3(const std::string &name, const glm::vec3 &value) const
{
    glUniform3fv(glGetUniformLocation(shaderID, name.c_str()), 1, &value[0]);
}
void Shader::SetVec3(const std::string &name, float x, float y, float z) const
{
    glUniform3f(glGetUniformLocation(shaderID, name.c_str()), x, y, z);
}

void Shader::SetVec4(const std::string &name, const glm::vec4 &value) const
{
    glUniform4fv(glGetUniformLocation(shaderID, name.c_str()), 1, &value[0]);
}
void Shader::SetVec4(const std::string &name, float x, float y, float z, float w)
{
    glUniform4f(glGetUniformLocation(shaderID, name.c_str()), x, y, z, w);
}

void Shader::SetMat2(const std::string &name, const glm::mat2 &mat) const
{
    glUniformMatrix2fv(glGetUniformLocation(shaderID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
}

void Shader::SetMat3(const std::string &name, const glm::mat3 &mat) const
{
    glUniformMatrix3fv(glGetUniformLocation(shaderID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
}

void Shader::SetMat4(const std::string &name, const glm::mat4 &mat) const
{
    glUniformMatrix4fv(glGetUniformLocation(shaderID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
}

#pragma endregion