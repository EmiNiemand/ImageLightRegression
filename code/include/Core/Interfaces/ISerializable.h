#ifndef IMAGELIGHTREGRESSION_ISERIALIZABLE_H
#define IMAGELIGHTREGRESSION_ISERIALIZABLE_H

#include "nlohmann/json.hpp"

class Object;

class ISerializable {
public:
    virtual void Save(nlohmann::json& json) = 0;
    virtual void Load(nlohmann::json& json) = 0;
};

#endif //IMAGELIGHTREGRESSION_ISERIALIZABLE_H
