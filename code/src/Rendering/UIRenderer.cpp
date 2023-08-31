#include "Rendering/UIRenderer.h"
#include "Managers/ResourceManager.h"
#include "Resources/Shader.h"
#include "Core/Object.h"

UIRenderer::UIRenderer() {
    imageShader = ResourceManager::LoadResource<Shader>("resources/Resources/ShaderResources/ImageShader.json");

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Rectangle::vertices), &Rectangle::vertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
}

UIRenderer::~UIRenderer() {
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);

    imageShader->Delete();
    ResourceManager::UnloadResource(imageShader->GetPath());
}

unsigned int UIRenderer::GetVAO() const {
    return vao;
}