#include "Resources/CubeMap.h"
#include "Macros.h"

#include "stb_image.h"
#include "glad/glad.h"
#include "nlohmann/json.hpp"

#include <fstream>

CubeMap::CubeMap(const std::string &inPath) : Resource(inPath) {}

CubeMap::~CubeMap() {
    glDeleteTextures(1, &id);
}

void CubeMap::Load() {
#ifdef RELEASE
    std::ifstream input(path);

    nlohmann::json json;
    input >> json;

    json.at("textureRight").get_to(textures[0]);
    json.at("textureLeft").get_to(textures[1]);
    json.at("textureTop").get_to(textures[2]);
    json.at("textureBottom").get_to(textures[3]);
    json.at("textureFront").get_to(textures[4]);
    json.at("textureBack").get_to(textures[5]);
#endif
#ifdef DEBUG
    try {
        std::ifstream input(path);

        nlohmann::json json;
        input >> json;

        json.at("textureRight").get_to(textures[0]);
        json.at("textureLeft").get_to(textures[1]);
        json.at("textureTop").get_to(textures[2]);
        json.at("textureBottom").get_to(textures[3]);
        json.at("textureFront").get_to(textures[4]);
        json.at("textureBack").get_to(textures[5]);
    }
    catch (std::ifstream::failure& e) {
        ILR_ERROR_MSG("Cubemap file path does not exist: " + path);
    }
#endif

    stbi_set_flip_vertically_on_load(false);
    unsigned int textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);

    int width, height, nrComponents;
    for (unsigned int i = 0; i < 6; i++)
    {
        unsigned char *data = stbi_load(textures[i].c_str(), &width, &height, &nrComponents, 0);
        if (data)
        {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
            stbi_image_free(data);
        }
        else
        {
            ILR_ERROR_MSG("Cubemap file path does not exist: " + textures[i]);
            stbi_image_free(data);
        }
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    stbi_set_flip_vertically_on_load(true);

    id = textureID;
}

unsigned int CubeMap::GetID() const {
    return id;
}

void CubeMap::Reload() {
    glDeleteTextures(1, &id);

    stbi_set_flip_vertically_on_load(false);
    unsigned int textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);

    int width, height, nrComponents;
    for (unsigned int i = 0; i < 6; i++)
    {
        unsigned char *data = stbi_load(textures[i].c_str(), &width, &height, &nrComponents, 0);
        if (data)
        {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
            stbi_image_free(data);
        }
        else
        {
            ILR_ERROR_MSG("Cubemap file path does not exist: " + textures[i]);
            stbi_image_free(data);
        }
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    stbi_set_flip_vertically_on_load(true);

    id = textureID;
}
