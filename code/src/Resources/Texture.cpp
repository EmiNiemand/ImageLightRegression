#include "Resources/Texture.h"
#include "Macros.h"

#include "stb_image.h"
#include "glad/glad.h"

Texture::Texture(const std::string &inPath) : Resource(inPath) {}

Texture::~Texture() {
    glDeleteTextures(1, &id);
}

void Texture::Load() {
    std::string filename = std::string(path);
    unsigned int textureID;

    glGenTextures(1, &textureID);

    int width, height, nrComponents;
    unsigned char *data = stbi_load(filename.c_str(), &width, &height, &nrComponents, 0);
    if (data) {
        GLenum format;
        if (nrComponents == 1)
            format = GL_RED;
        else if (nrComponents == 3)
            format = GL_RGB;
        else if (nrComponents == 4)
            format = GL_RGBA;

        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, (GLint)format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    } else {
        ILR_ERROR_MSG("Texture failed to load at path: " + (std::string)path);
    }
    stbi_image_free(data);
    id = textureID;
}

unsigned int Texture::GetID() const {
    return id;
}
