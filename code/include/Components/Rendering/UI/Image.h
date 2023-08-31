#ifndef IMAGELIGHTREGRESSION_IMAGE_H
#define IMAGELIGHTREGRESSION_IMAGE_H

#include "Components/Component.h"

#include "glm/glm.hpp"

#include <string>

class Texture;
class Shader;

class Image : public Component {
public:
    glm::vec2 size = {1.0f, 1.0f};

private:
    Texture* texture = nullptr;

public:
    Image(Object *parent, int id);
    ~Image() override;

    void OnDestroy() override;

    void Draw(Shader* inShader);
    /**
     *
     * @param id - texture id
     * @param inShader - pointer to a shader object
     * @param inSize - normalized size from 0 to 1, where 1 is value for full screen textures
     * @param inPosition - position on the screen, (0, 0) is down left corner
     * @param inPivot - pivot value from 0 to 1;
     */
    static void DrawImageByID(unsigned int id, Shader *inShader, const glm::vec2& inSize,
                              const glm::vec2& inPosition = glm::vec2(0.0f), const glm::vec2& inPivot = glm::vec2(0.5f));

    void SetTexture(const std::string& inPath);
    [[nodiscard]] unsigned int GetTextureID();
};


#endif //IMAGELIGHTREGRESSION_IMAGE_H
