#ifndef NNET_NN_CHECKPOINT_H
#define NNET_NN_CHECKPOINT_H

#include "nnet/nn/module.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// File layout: magic number, version, parameter count, then each parameter's
// name, shape and values. Only the numbers are saved; the model's structure
// comes from the code that builds it.

namespace nnet {

    namespace checkpoint_detail {
        constexpr std::uint64_t magicNumber = 0x4E4E4554;  // "NNET"
        constexpr std::uint64_t formatVersion = 1;

        inline void writeInteger(std::ofstream& file, std::uint64_t value) {
            file.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }

        inline std::uint64_t readInteger(std::ifstream& file, const std::string& path) {
            std::uint64_t value = 0;
            if (!file.read(reinterpret_cast<char*>(&value), sizeof(value))) {
                throw std::runtime_error(path + ": checkpoint ends early");
            }
            return value;
        }
    }

    inline void saveParameters(Module& model, const std::string& path) {
        using namespace checkpoint_detail;
        std::ofstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error(path + ": could not open for writing");
        }

        const std::vector<NamedParameter> parameters = model.namedParameters();
        writeInteger(file, magicNumber);
        writeInteger(file, formatVersion);
        writeInteger(file, parameters.size());
        for (const NamedParameter& entry : parameters) {
            const Tensor& value = entry.parameter->value;
            writeInteger(file, entry.name.size());
            file.write(entry.name.data(), static_cast<std::streamsize>(entry.name.size()));
            writeInteger(file, value.rank());
            for (const std::size_t dimension : value.shape()) {
                writeInteger(file, dimension);
            }
            file.write(reinterpret_cast<const char*>(value.getData().data()),
                       static_cast<std::streamsize>(value.numel() * sizeof(double)));
        }

        file.close();
        if (!file) {
            throw std::runtime_error(path + ": could not write checkpoint");
        }
    }

    // The model must be built the same way as the one that was saved.
    inline void loadParameters(Module& model, const std::string& path) {
        using namespace checkpoint_detail;
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error(path + ": could not open checkpoint");
        }
        if (readInteger(file, path) != magicNumber) {
            throw std::runtime_error(path + ": not a checkpoint file");
        }
        if (readInteger(file, path) != formatVersion) {
            throw std::runtime_error(path + ": unsupported checkpoint version");
        }

        const std::vector<NamedParameter> parameters = model.namedParameters();
        if (readInteger(file, path) != parameters.size()) {
            throw std::runtime_error(path + ": checkpoint has a different number of parameters than the model");
        }

        // read and check everything before touching the model
        std::vector<Tensor> loadedValues;
        for (const NamedParameter& entry : parameters) {
            const std::string mismatch = path + ": checkpoint doesn't match the model at '" + entry.name + "'";

            const std::uint64_t nameLength = readInteger(file, path);
            if (nameLength != entry.name.size()) {
                throw std::runtime_error(mismatch);
            }
            std::string name(nameLength, ' ');
            if (!file.read(name.data(), static_cast<std::streamsize>(nameLength))) {
                throw std::runtime_error(path + ": checkpoint ends early");
            }
            if (name != entry.name) {
                throw std::runtime_error(mismatch);
            }

            const Tensor::Shape& shape = entry.parameter->value.shape();
            if (readInteger(file, path) != shape.size()) {
                throw std::runtime_error(mismatch);
            }
            for (const std::size_t dimension : shape) {
                if (readInteger(file, path) != dimension) {
                    throw std::runtime_error(mismatch);
                }
            }

            std::vector<double> values(entry.parameter->value.numel());
            if (!file.read(reinterpret_cast<char*>(values.data()),
                           static_cast<std::streamsize>(values.size() * sizeof(double)))) {
                throw std::runtime_error(path + ": checkpoint ends early");
            }
            loadedValues.push_back(Tensor(shape, std::move(values)));
        }
        if (file.peek() != std::ifstream::traits_type::eof()) {
            throw std::runtime_error(path + ": unexpected data after the last parameter");
        }

        for (std::size_t i = 0; i < parameters.size(); i++) {
            parameters[i].parameter->value = std::move(loadedValues[i]);
        }
    }

}

#endif
