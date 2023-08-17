#ifndef IMAGELIGHTREGRESSION_FILEEXPLORER_H
#define IMAGELIGHTREGRESSION_FILEEXPLORER_H

#include <string>

class FileExplorer {
public:
    static void ShowFiles();

private:
    static void ShowDirectory(const std::string& path);
};


#endif //IMAGELIGHTREGRESSION_FILEEXPLORER_H
