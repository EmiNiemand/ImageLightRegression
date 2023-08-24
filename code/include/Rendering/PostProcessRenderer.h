#ifndef IMAGELIGHTREGRESSION_POSTPROCESSRENDERER_H
#define IMAGELIGHTREGRESSION_POSTPROCESSRENDERER_H

class Shader;

class PostProcessRenderer {
public:
    Shader* postProcessShader = nullptr;

private:
    unsigned int vao = 0;
    unsigned int vbo = 0;

public:
    PostProcessRenderer();
    virtual ~PostProcessRenderer();

    void Draw() const;
};


#endif //IMAGELIGHTREGRESSION_POSTPROCESSRENDERER_H
