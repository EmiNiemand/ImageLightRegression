#ifndef IMAGELIGHTREGRESSION_CUBEMAP_H
#define IMAGELIGHTREGRESSION_CUBEMAP_H

#include "Core/Resource.h"

class CubeMap : public Resource {
public:
    std::string textures[6];

private:
    unsigned int id = 0;

public:
    explicit CubeMap(const std::string &inPath);
    ~CubeMap() override;

    void Load() override;

    [[nodiscard]] unsigned int GetID() const;
};


#endif //IMAGELIGHTREGRESSION_CUBEMAP_H
