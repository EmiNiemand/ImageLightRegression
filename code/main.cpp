#include "Application.h"

int main(int, char**)
{
    Application::GetInstance()->StartUp();

    Application::GetInstance()->Run();

    Application::GetInstance()->ShutDown();
    return 0;
}
