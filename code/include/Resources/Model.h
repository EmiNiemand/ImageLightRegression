#ifndef IMAGELIGHTREGRESSION_MODEL_H
#define IMAGELIGHTREGRESSION_MODEL_H

#include "Core/Resource.h"
#include "Mesh.h"
#include "assimp/scene.h"

class Model : public Resource {
private:
    // model data
    std::vector<Mesh> meshes;

public:
    explicit Model(const std::string &inPath);
    ~Model() override;

    void Load() override;

    void Draw(Shader* shader);

private:
    void ProcessNode(aiNode *node, const aiScene *scene);
    Mesh ProcessMesh(aiMesh *mesh, const aiScene *scene);
    std::vector<Texture*> LoadMaterialTextures(aiMaterial *mat, aiTextureType type, const std::string& typeName);
};


#endif //IMAGELIGHTREGRESSION_MODEL_H
