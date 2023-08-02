#ifndef IMAGELIGHTREGRESSION_TEXTURE_H
#define IMAGELIGHTREGRESSION_TEXTURE_H

#include "Core/Resource.h"

class Texture : public Resource {
public:
    std::string type = {};

private:
    unsigned int id = 0;

public:
    explicit Texture(const std::string &inPath);
    ~Texture() override;

    void Load() override;
    unsigned int GetID() const;
};


#endif //IMAGELIGHTREGRESSION_TEXTURE_H
