#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>

int main(int argc, char **argv)
{
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <sourceFile> <headerFile> <arrayName>" << std::endl;
        return 1;
    }

    std::string sourceFilename(argv[1]);
    std::string headerFilename(argv[2]);
    std::string arrayName(argv[3]);

    std::ifstream fin(sourceFilename, std::ios::binary);
    std::ofstream fout(headerFilename, std::ios_base::app);

    if (!fin) {
        std::cerr << "Can't open file " << sourceFilename << "!" << std::endl;
        return 1;
    }

    // #include <cstddef>
    //
    // static const char <arrayName>[] = {
    //  /*... hexadecimal data from source file ...*/
    // };
    //
    // size_t <arrayName>_length = sizeof(<arrayName>) / sizeof(char);

    fout << "#include <cstddef>" << std::endl;
    fout << std::endl;
    fout << "static const char " << arrayName << "[] = {" << std::endl;

    char buffer[2391];
    const int maxBytesInLine = 120 / 6;
    int bytesInLine = 0;

    std::streamsize n;
    do {
        fin.read(buffer, sizeof(buffer) / sizeof(char));
        n = fin.gcount();
        for (std::streamsize i = 0; i < n; ++i) {
            unsigned int value = (unsigned int) buffer[i];
            if (value > 0xff) {
                value -= 0xffffff00;
            }
            if (value >= 128) {
                fout << "-";
                value = 256 - value;
            }
            fout << "0x" << std::setw(2) << std::setfill('0') << std::hex << value << ", ";
            ++bytesInLine;
            if (bytesInLine == maxBytesInLine) {
                fout << std::endl;
                bytesInLine = 0;
            }
        }
    } while (n > 0);

    if (bytesInLine > 0)
        fout << std::endl;

    fout << "};" << std::endl;
    fout << std::endl;
    fout << "size_t " << arrayName << "_length = sizeof(" << arrayName << ") / sizeof(char);" << std::endl;

    fin.close();
    fout.close();

    return 0;
}
