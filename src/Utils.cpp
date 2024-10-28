#include "Utils.h"
#include <cstdlib>
#include "Types.h"

namespace ML {

    // void quantize_to_8bit_file(char* filePath, bool bias, fp32 max)
    // {

    //     std::ifstream file(filePath, std::ios::binary);  // Open our file
    //     if (file.is_open()) {
    //         std::cout << "Opened binary file " << filePath << std::endl;
    //     } else {
    //         std::cout << "Failed to open binary file " << filePath << std::endl;
    //         return;
    //     }

    //     std::vector<float> buffer;
    //     float value;
    //     while (file.read(reinterpret_cast<char*>(&value), sizeof(value))) {
    //         buffer.push_back(value);
    //     }

    //     if (!file.eof()) {
    //         std::cerr << "Error reading file" << std::endl;
    //         return;
    //     }

    //     // Process the floats (for example, print them)
    //     for (const float &f : buffer) {
    //         //Quantize floats
    //         if(weight)
    //         {
    //             fp32 Sw = 127.0 / max;
    //             int8_t Wx = round(Sw * f);

    //         }
    //         if(bias)
    //         {
    //             fp32 Si = 127.0 / std::max()
    //         }
    //     }

    // }

}  // namespace ML