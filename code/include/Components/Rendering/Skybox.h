#ifndef IMAGELIGHTREGRESSION_SKYBOX_H
#define IMAGELIGHTREGRESSION_SKYBOX_H

#include "Components/Component.h"
#include "Structures.h"

class Object;
class CubeMap;
class Shader;

class Skybox : public Component {
private:
    CubeMap* cubeMap = nullptr;

public:
    Skybox(Object *parent, int id);
    ~Skybox() override;

    void OnDestroy() override;

    [[nodiscard]] CubeMap* GetCubeMap() const;

    void SetCubeMap(std::string cubeMapPath);
    void SetActive();

    void Save(nlohmann::json &json) override;
    void Load(nlohmann::json &json) override;
};


#endif //IMAGELIGHTREGRESSION_SKYBOX_H
