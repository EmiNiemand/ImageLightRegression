#ifndef IMAGELIGHTREGRESSION_CUM_H
#define IMAGELIGHTREGRESSION_CUM_H

#include "ApplicationTypes.h"
#include "Structures.h"

#include "nlohmann/json.hpp"

#include <string>

class CUM {
public:
    static uint64 Hash(const std::string& path);
    static bool IsInViewport(glm::ivec2 position, Viewport* viewport);
    static void SaveJsonToFile(const std::string& filePath, const nlohmann::json& json);
    static void LoadJsonFromFile(const std::string& filePath, nlohmann::json& json);
};


#endif //IMAGELIGHTREGRESSION_CUM_H
