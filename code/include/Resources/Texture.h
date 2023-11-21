#ifndef IMAGELIGHTREGRESSION_TEXTURE_H
#define IMAGELIGHTREGRESSION_TEXTURE_H

#include "Core/Resource.h"

#include "glm/glm.hpp"

class Texture : public Resource {
public:
    std::string type = {};

private:
    unsigned int id = 0;
    glm::ivec2 resolution = {};

public:
    explicit Texture(const std::string &inPath);
    ~Texture() override;

    void Load() override;

    void SetID(unsigned int inID);
    void SetResolution(const glm::ivec2& inResolution);

    [[nodiscard]] unsigned int GetID() const;
    [[nodiscard]] const glm::ivec2& GetResolution() const;
};


#endif //IMAGELIGHTREGRESSION_TEXTURE_H
