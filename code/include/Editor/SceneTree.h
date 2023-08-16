#ifndef IMAGELIGHTREGRESSION_SCENETREE_H
#define IMAGELIGHTREGRESSION_SCENETREE_H

class Object;

class SceneTree {
public:
    static void ShowTreeNode(Object* parent);
    static void ShowPopUp();

private:
    static void ManageNodeInput(Object* hoveredObject);
};


#endif //IMAGELIGHTREGRESSION_SCENETREE_H
