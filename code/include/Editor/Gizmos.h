#ifndef IMAGELIGHTREGRESSION_GIZMOS_H
#define IMAGELIGHTREGRESSION_GIZMOS_H

#include "ApplicationTypes.h"

#include "glm/glm.hpp"

#include <vector>

class Shader;

class Gizmos {
public:
    Shader* gizmoShader = nullptr;
    int8 hookedPoint = -1;
    std::vector<glm::vec2> hookPoints;

    int mode = 0;
private:
    double cursorPreviousX = -1.0f;
    double cursorPreviousY = -1.0f;

    unsigned int vao = 0;
    unsigned int vbo = 0;

public:
    Gizmos();
    virtual ~Gizmos();

    void Draw();
    void Update();
};


#endif //IMAGELIGHTREGRESSION_GIZMOS_H
