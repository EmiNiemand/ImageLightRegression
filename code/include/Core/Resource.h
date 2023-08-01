#ifndef IMAGELIGHTREGRESSION_RESOURCE_H
#define IMAGELIGHTREGRESSION_RESOURCE_H

#include "ApplicationTypes.h"
#include <filesystem>

class Resource {
private:
    std::string path;

public:
    explicit Resource(std::string inPath);
    virtual ~Resource();
    virtual void Load() = 0;
    void Unload();
};


#endif //IMAGELIGHTREGRESSION_RESOURCE_H
