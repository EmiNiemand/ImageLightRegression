#ifndef IMAGELIGHTREGRESSION_TOOLBAR_H
#define IMAGELIGHTREGRESSION_TOOLBAR_H

#include <string>

class ToolBar {
public:
    static void ShowToolBar();
private:
    static void ShowButton(const std::string& label, unsigned int textureID, bool isActive = true);
    static void SaveRenderToFile(const std::string& path);
};


#endif //IMAGELIGHTREGRESSION_TOOLBAR_H
