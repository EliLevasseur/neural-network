#include "nnet/data/mnist.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nnet {

    namespace {
        constexpr std::uint32_t imageMagicNumber = 2051;
        constexpr std::uint32_t labelMagicNumber = 2049;
        constexpr std::size_t imageSide = 28;
        constexpr std::size_t digitCount = 10;

        // Reads all of a file's bytes, refusing to start if the file is
        // missing.
        std::vector<unsigned char> readWholeFile(const std::string& path) {
            std::ifstream file(path, std::ios::binary);
            if (!file) {
                throw std::runtime_error(path + ": could not open file");
            }
            const std::uintmax_t size = std::filesystem::file_size(path);
            std::vector<unsigned char> bytes(static_cast<std::size_t>(size));
            if (!file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size))) {
                throw std::runtime_error(path + ": could not read file");
            }
            return bytes;
        }

        // The header number that starts at `offset`, put back together from
        // its four big-endian bytes.
        std::uint32_t headerNumber(const std::vector<unsigned char>& bytes, std::size_t offset,
                                   const std::string& path) {
            if (bytes.size() < offset + 4) {
                throw std::runtime_error(path + ": file ends inside its header");
            }
            return (static_cast<std::uint32_t>(bytes[offset]) << 24) |
                   (static_cast<std::uint32_t>(bytes[offset + 1]) << 16) |
                   (static_cast<std::uint32_t>(bytes[offset + 2]) << 8) |
                   static_cast<std::uint32_t>(bytes[offset + 3]);
        }

        void expectSize(const std::vector<unsigned char>& bytes, std::size_t expected,
                        const std::string& path) {
            if (bytes.size() != expected) {
                throw std::runtime_error(path + ": file is " + std::to_string(bytes.size()) +
                    " bytes, but its header describes " + std::to_string(expected));
            }
        }
    }

    MnistData loadMnist(const std::string& imagePath, const std::string& labelPath) {
        const std::vector<unsigned char> imageBytes = readWholeFile(imagePath);
        if (headerNumber(imageBytes, 0, imagePath) != imageMagicNumber) {
            throw std::runtime_error(imagePath + ": not an IDX image file (wrong magic number)");
        }
        const std::size_t imageCount = headerNumber(imageBytes, 4, imagePath);
        if (headerNumber(imageBytes, 8, imagePath) != imageSide ||
            headerNumber(imageBytes, 12, imagePath) != imageSide) {
            throw std::runtime_error(imagePath + ": images are not 28x28");
        }
        const std::size_t imageHeaderSize = 16;
        const std::size_t pixelsPerImage = imageSide * imageSide;
        expectSize(imageBytes, imageHeaderSize + imageCount * pixelsPerImage, imagePath);

        const std::vector<unsigned char> labelBytes = readWholeFile(labelPath);
        if (headerNumber(labelBytes, 0, labelPath) != labelMagicNumber) {
            throw std::runtime_error(labelPath + ": not an IDX label file (wrong magic number)");
        }
        const std::size_t labelCount = headerNumber(labelBytes, 4, labelPath);
        const std::size_t labelHeaderSize = 8;
        expectSize(labelBytes, labelHeaderSize + labelCount, labelPath);

        if (imageCount != labelCount) {
            throw std::runtime_error("MNIST image and label files hold different numbers of examples");
        }
        if (imageCount == 0) {
            throw std::runtime_error(imagePath + ": holds no examples");
        }

        std::vector<double> imageValues(imageCount * pixelsPerImage);
        for (std::size_t i = 0; i < imageValues.size(); i++) {
            imageValues[i] = static_cast<double>(imageBytes[imageHeaderSize + i]) / 255.0;
        }

        std::vector<double> labelValues(labelCount * digitCount, 0.0);
        for (std::size_t example = 0; example < labelCount; example++) {
            const std::size_t digit = labelBytes[labelHeaderSize + example];
            if (digit >= digitCount) {
                throw std::runtime_error(labelPath + ": label " + std::to_string(digit) + " is not a digit");
            }
            labelValues[example * digitCount + digit] = 1.0;
        }

        return MnistData{Tensor({imageCount, pixelsPerImage}, std::move(imageValues)),
                         Tensor({labelCount, digitCount}, std::move(labelValues))};
    }

}
