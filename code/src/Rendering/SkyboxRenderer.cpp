#include "Rendering/SkyboxRenderer.h"
#include "Managers/ResourceManager.h"
#include "Components/Rendering/Skybox.h"
#include "Core/Object.h"
#include "Resources/CubeMap.h"
#include "Resources/Shader.h"

SkyboxRenderer::SkyboxRenderer() {
    cubeMapShader = ResourceManager::LoadResource<Shader>("resources/Resources/ShaderResources/CubeMapShader.json");

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Cube::vertices), &Cube::vertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    cubeMapShader->Activate();
    cubeMapShader->SetInt("cubeMapTexture", 4);
}

SkyboxRenderer::~SkyboxRenderer() {
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);

    cubeMapShader->Delete();
    ResourceManager::UnloadResource(cubeMapShader->GetPath());
}

void SkyboxRenderer::SetActiveSkybox(Object* inSkybox) {
    activeSkybox = inSkybox;
}


Object *SkyboxRenderer::GetActiveSkybox() const {
    return activeSkybox;
}

void SkyboxRenderer::Draw() {
    if (!activeSkybox || !activeSkybox->GetComponentByClass<Skybox>()->GetEnabled()) return;

    glDepthFunc(GL_LEQUAL);
    cubeMapShader->Activate();

    glBindVertexArray(vao);
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_CUBE_MAP, activeSkybox->GetComponentByClass<Skybox>()->GetCubeMap()->GetID());

    glDrawArrays(GL_TRIANGLES, 0, 36);
    glBindVertexArray(0);

    glActiveTexture(GL_TEXTURE0);

    glDepthFunc(GL_LESS);

}
