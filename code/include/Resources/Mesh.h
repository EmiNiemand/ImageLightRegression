#ifndef IMAGELIGHTREGRESSION_MESH_H
#define IMAGELIGHTREGRESSION_MESH_H

#include "Structures.h"

#include <vector>

class Shader;
class Texture;

class Mesh {
public:
    // mesh Data
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    std::vector<Texture*> textures;
    unsigned int vao;

private:
    // render data
    unsigned int vbo, ebo;

public:
    Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, std::vector<Texture*> textures);
    virtual ~Mesh();

    // render the mesh
    void Draw(Shader* inShader);

private:
    // initializes all the buffer objects/arrays
    void InitializeBuffers();
};


#endif //IMAGELIGHTREGRESSION_MESH_H
