#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

extern "C"
int response(double I1, double I2, double& dpsi1, double& dpsi2)
{
    std::system(("python3 din_lib_anl.py "  + std::to_string(I1) + " " + std::to_string(I2)).c_str()); // executes the UNIX command "ls -l >test.txt"

    std::ifstream f("file.txt");
    f >> dpsi1 >> dpsi2;
    
    return 0;
 
}

// int main()
// {
//    double dpsi1 = 1.0, dpsi2 = 1.0;
//    response(1, 2, dpsi1, dpsi2);

//    return 0;
// }
