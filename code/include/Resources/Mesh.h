#ifndef IMAGELIGHTREGRESSION_MESH_H
#define IMAGELIGHTREGRESSION_MESH_H

#define MAX_BONE_INFLUENCE 4

#include "glm/glm.hpp"
#include <vector>

class Shader;
class Texture;

struct Vertex {
    // position
    glm::vec3 position;
    // normal
    glm::vec3 normal;
    // texCoords
    glm::vec2 texCoords;
    // tangent
    glm::vec3 tangent;
    // bitangent
    glm::vec3 biTangent;
    //bone indexes which will influence this vertex
    //glm::i16vec4 boneIDs[MAX_BONE_INFLUENCE];
    int boneIDs[MAX_BONE_INFLUENCE];
    //weights from each bone
    //glm::vec4 weights[MAX_BONE_INFLUENCE];
    float weights[MAX_BONE_INFLUENCE];
};

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
    // constructor
    Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, std::vector<Texture*> textures);
    virtual ~Mesh();

    // render the mesh
    void Draw(Shader* shader);

private:
    // initializes all the buffer objects/arrays
    void setupMesh();
};


#endif //IMAGELIGHTREGRESSION_MESH_H
