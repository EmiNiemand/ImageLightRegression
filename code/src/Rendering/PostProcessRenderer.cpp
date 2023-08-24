#include "Rendering/PostProcessRenderer.h"
#include "Managers/ResourceManager.h"
#include "Resources/Shader.h"
#include "Structures.h"

PostProcessRenderer::PostProcessRenderer() {
    postProcessShader = ResourceManager::LoadResource<Shader>("resources/Resources/ShaderResources/PostProcessShader.json");

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Rectangle::vertices), &Rectangle::vertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));

    postProcessShader->Activate();
    postProcessShader->SetInt("screenTexture", 0);
    postProcessShader->SetInt("selectedObjectTexture", 1);
}

PostProcessRenderer::~PostProcessRenderer() {
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);

    postProcessShader->Delete();
    ResourceManager::UnloadResource(postProcessShader->GetPath());
}

void PostProcessRenderer::Draw() const {
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
}


