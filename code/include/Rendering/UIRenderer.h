#ifndef IMAGELIGHTREGRESSION_UIRENDERER_H
#define IMAGELIGHTREGRESSION_UIRENDERER_H

class Shader;

class UIRenderer {
public:
    Shader* imageShader = nullptr;

private:
    unsigned int vao = 0;
    unsigned int vbo = 0;

public:
    UIRenderer();
    virtual ~UIRenderer();

    [[nodiscard]] unsigned int GetVAO() const;
};


#endif //IMAGELIGHTREGRESSION_UIRENDERER_H
