#ifndef IMAGELIGHTREGRESSION_CUM_H
#define IMAGELIGHTREGRESSION_CUM_H

#include "ApplicationTypes.h"
#include "Structures.h"

#include <string>

class CUM {
public:
    static uint64 Hash(const std::string& path);
    static bool IsInViewport(glm::ivec2 position, Viewport* viewport);
};


#endif //IMAGELIGHTREGRESSION_CUM_H
