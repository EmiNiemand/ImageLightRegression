#ifndef IMAGELIGHTREGRESSION_RESOURCE_H
#define IMAGELIGHTREGRESSION_RESOURCE_H

#include "ApplicationTypes.h"
#include <filesystem>

class Resource {
protected:
    std::string path;

public:
    explicit Resource(std::string inPath);
    virtual ~Resource();

    const std::string& GetPath();

protected:
    virtual void Load() = 0;

private:
    void Unload();
};


#endif //IMAGELIGHTREGRESSION_RESOURCE_H
