#ifndef IMAGELIGHTREGRESSION_EDITORMANAGER_H
#define IMAGELIGHTREGRESSION_EDITORMANAGER_H


class EditorManager {
private:
    inline static EditorManager* editorManager;

    // Decide GL+GLSL versions
    // GL 4.3 + GLSL 430
    const char* glsl_version = "#version 430";

public:
    EditorManager(EditorManager &other) = delete;
    void operator=(const EditorManager&) = delete;
    virtual ~EditorManager();

    static EditorManager* GetInstance();

    void Startup();
    void Shutdown();

    void Show();

private:
    explicit EditorManager();
};


#endif //IMAGELIGHTREGRESSION_EDITORMANAGER_H
