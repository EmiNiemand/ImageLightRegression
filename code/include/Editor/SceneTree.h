#ifndef IMAGELIGHTREGRESSION_SCENETREE_H
#define IMAGELIGHTREGRESSION_SCENETREE_H

class Object;

class SceneTree {
private:
    inline static SceneTree* sceneTree;

public:
    SceneTree(SceneTree &other) = delete;
    void operator=(const SceneTree&) = delete;
    virtual ~SceneTree();

    static SceneTree* GetInstance();

    void ShowTreeNode(Object* parent);
    void ShowPopUp();

private:
    explicit SceneTree();

    void ManageNodeInput(Object* hoveredObject);
};


#endif //IMAGELIGHTREGRESSION_SCENETREE_H
