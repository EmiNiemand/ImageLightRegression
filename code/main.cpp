#include "Application.h"

int main(int, char**)
{
    Application::GetInstance()->Startup();

    Application::GetInstance()->Run();

    Application::GetInstance()->Shutdown();
    return 0;
}
