#include "Factories/ComponentFactory.h"
#include "Core/Object.h"

ComponentFactory::ComponentFactory() = default;

ComponentFactory::~ComponentFactory() {
    delete componentFactory;
}

ComponentFactory* ComponentFactory::GetInstance() {
    if (componentFactory == nullptr) {
        componentFactory = new ComponentFactory();
    }
    return componentFactory;
}
