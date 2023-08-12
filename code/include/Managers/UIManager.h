#ifndef IMAGELIGHTREGRESSION_UIMANAGER_H
#define IMAGELIGHTREGRESSION_UIMANAGER_H

class Shader;

class UIManager {
public:
    Shader* imageShader = nullptr;

private:
    inline static UIManager* uiManager;
    unsigned int vao = 0;
    unsigned int vbo = 0;

public:
    UIManager(UIManager &other) = delete;
    void operator=(const UIManager&) = delete;
    virtual ~UIManager();

    static UIManager* GetInstance();

    void Startup();
    void Shutdown();

    void UpdateProjection() const;

    [[nodiscard]] unsigned int GetVAO() const;

private:
    explicit UIManager();

};


#endif //IMAGELIGHTREGRESSION_UIMANAGER_H
