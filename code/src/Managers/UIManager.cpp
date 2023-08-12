#include "Managers/UIManager.h"
#include "Managers/ResourceManager.h"
#include "Resources/Shader.h"
#include "Core/Object.h"
#include "Components/Rendering/Camera.h"

UIManager::UIManager() = default;

UIManager::~UIManager() = default;

UIManager *UIManager::GetInstance() {
    if (uiManager == nullptr) {
        uiManager = new UIManager();
    }
    return uiManager;
}

void UIManager::UpdateProjection() const {
    glm::mat4 projection = Camera::GetActiveCamera()->GetComponentByClass<Camera>()->GetProjectionMatrix();

    imageShader->Activate();
    imageShader->SetMat4("projection", projection);
}

void UIManager::Startup() {
    imageShader = ResourceManager::LoadResource<Shader>("resources/Resources/ShaderResources/ImageShader.json");

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Rectangle::vertices), &Rectangle::vertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

}

void UIManager::Shutdown() {
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);

    imageShader->Delete();
    ResourceManager::UnloadResource(imageShader->GetPath());

    delete uiManager;
}

unsigned int UIManager::GetVAO() const {
    return vao;
}
