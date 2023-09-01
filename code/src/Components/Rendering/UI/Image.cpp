#include "Components/Rendering/UI/Image.h"
#include "Managers/ResourceManager.h"
#include "Managers/RenderingManager.h"
#include "Rendering/UIRenderer.h"
#include "Core/Object.h"
#include "Components/Transform.h"
#include "Resources/Texture.h"
#include "Resources/Shader.h"

#include <glad/glad.h>

Image::Image(Object *parent, int id) : Component(parent, id) {
    texture = ResourceManager::LoadResource<Texture>("resources/Textures/DefaultImage.png");

    Shader* shader = RenderingManager::GetInstance()->uiRenderer->imageShader;
    shader->Activate();
    shader->SetInt("imageTexture", 0);
}

Image::~Image() = default;

void Image::OnDestroy() {
    Component::OnDestroy();

    ResourceManager::UnloadResource(texture->GetPath());
}

void Image::Draw(Shader *inShader) {
    if (!parent->GetEnabled()) return;

    inShader->Activate();
    inShader->SetVec2("size", size);
    inShader->SetVec2("screenPosition", glm::vec2(parent->transform->GetGlobalPosition()));
    inShader->SetVec2("pivot", glm::vec2(0.5f, 0.5f));

    glBindVertexArray(RenderingManager::GetInstance()->uiRenderer->GetVAO());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture->GetID());

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
}

void Image::DrawImageByID(unsigned int id, Shader *inShader, const glm::vec2 &inSize, const glm::vec2 &inPosition,
                          const glm::vec2 &inPivot) {

    inShader->Activate();
    inShader->SetVec2("size", inSize);
    inShader->SetVec2("screenPosition", inPosition);
    inShader->SetVec2("pivot", inPivot);

    glBindVertexArray(RenderingManager::GetInstance()->uiRenderer->GetVAO());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, id);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
}

void Image::SetTexture(const std::string &inPath) {
    if (texture != nullptr) {
        ResourceManager::UnloadResource(texture->GetPath());
    }
    texture = ResourceManager::LoadResource<Texture>(inPath);
}

unsigned int Image::GetTextureID() {
    return texture->GetID();
}

void Image::Save(nlohmann::json &json) {
    Component::Save(json);

    json["ComponentType"] = "Image";

    json["Size"] = nlohmann::json::array();
    json["Size"].push_back(size.x);
    json["Size"].push_back(size.y);

    json["Texture"] = texture->GetPath();
}

void Image::Load(nlohmann::json &json) {
    Component::Load(json);

    size.x = json["Size"][0];
    size.y = json["Size"][1];

    SetTexture(json["Texture"]);
}


