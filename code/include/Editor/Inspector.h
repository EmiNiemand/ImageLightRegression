#ifndef IMAGELIGHTREGRESSION_INSPECTOR_H
#define IMAGELIGHTREGRESSION_INSPECTOR_H

#include <string>

class Inspector {
public:
    void ShowComponentProperties(const std::string& componentName);

private:
    void ShowTransform();
    void ShowCamera();
    void ShowRenderer();
    void ShowPointLight();
    void ShowSpotLight();
    void ShowDirectionalLight();

};


#endif //IMAGELIGHTREGRESSION_INSPECTOR_H
