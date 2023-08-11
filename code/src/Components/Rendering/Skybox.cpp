#include "Components/Rendering/Skybox.h"
#include "Managers/ResourceManager.h"
#include "Managers/RenderingManager.h"
#include "Core/Object.h"
#include "Resources/CubeMap.h"
#include "Resources/Shader.h"


Skybox::Skybox(Object *parent, int id) : Component(parent, id) {
    cubeMap = ResourceManager::LoadResource<CubeMap>("resources/Resources/CubeMap.json");

    Shader* shader = RenderingManager::GetInstance()->cubeMapShader;
    shader->Activate();
    shader->SetInt("cubeMapTexture", 4);
}

Skybox::~Skybox() = default;

void Skybox::Draw(Shader* inShader) {
    if (activeSkybox && !activeSkybox->GetComponentByClass<Skybox>()->enabled) return;
    // change depth function so depth test passes
    // when values are equal to depth buffer's content
    glDepthFunc(GL_LEQUAL);
    inShader->Activate();

    glBindVertexArray(vao);
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_CUBE_MAP, activeSkybox->GetComponentByClass<Skybox>()->cubeMap->GetID());

    glDrawArrays(GL_TRIANGLES, 0, 36);
    glBindVertexArray(0);

    glDepthFunc(GL_LESS); // set depth function back to default
}

void Skybox::SetActiveSkybox(Object* inSkybox) {
    activeSkybox = inSkybox;
}

void Skybox::InitializeBuffers() {
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Cube::vertices), &Cube::vertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
}

void Skybox::DeleteBuffers() {
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
}

void Skybox::OnDestroy() {
    Component::OnDestroy();
    ResourceManager::UnloadResource(cubeMap->GetPath());
}
