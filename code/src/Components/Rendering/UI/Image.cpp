#include "Components/Rendering/UI/Image.h"
#include "Managers/ResourceManager.h"
#include "Managers/RenderingManager.h"
#include "Core/Object.h"
#include "Components/Transform.h"
#include "Resources/Texture.h"
#include "Resources/Shader.h"
#include "Structures.h"

#include <glad/glad.h>

Image::Image(Object *parent, int id) : Component(parent, id) {
    texture = ResourceManager::LoadResource<Texture>("resources/Textures/DefaultImage.png");

    Shader* shader = RenderingManager::GetInstance()->imageShader;
    shader->Activate();
    shader->SetInt("imageTexture", 0);
}

Image::~Image() = default;

void Image::OnDestroy() {
    Component::OnDestroy();

    ResourceManager::UnloadResource(texture->GetPath());
}

void Image::Draw(Shader *inShader) {
    if (!parent->enabled) return;

    inShader->Activate();
    inShader->SetVec2("size", size);
    inShader->SetVec2("screenPosition", glm::vec2(parent->transform->GetGlobalPosition()));
    inShader->SetVec2("pivot", glm::vec2(0.5f, 0.5f));

    glBindVertexArray(vao);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture->GetID());

    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}

void Image::SetTexture(const std::string &inPath) {
    ResourceManager::UnloadResource(texture->GetPath());
    texture = ResourceManager::LoadResource<Texture>(inPath);
}

void Image::InitializeBuffers() {
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Rectangle::vertices), &Rectangle::vertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

}

void Image::DeleteBuffers() {
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
}

