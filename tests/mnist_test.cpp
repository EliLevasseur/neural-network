#include "test_utils.h"
#include "nnet/data/mnist.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// These tests write tiny IDX files of their own, so they do not need the real
// MNIST download (which is not committed).

namespace {
    void appendBigEndian(std::vector<unsigned char>& bytes, std::uint32_t number) {
        bytes.push_back(static_cast<unsigned char>(number >> 24));
        bytes.push_back(static_cast<unsigned char>(number >> 16));
        bytes.push_back(static_cast<unsigned char>(number >> 8));
        bytes.push_back(static_cast<unsigned char>(number));
    }

    // An image file holding `count` 28x28 images. Every pixel of image i is
    // `fill[i]`, except the first pixel, which is always 255.
    std::vector<unsigned char> imageFile(const std::vector<unsigned char>& fill, std::uint32_t magic = 2051) {
        std::vector<unsigned char> bytes;
        appendBigEndian(bytes, magic);
        appendBigEndian(bytes, static_cast<std::uint32_t>(fill.size()));
        appendBigEndian(bytes, 28);
        appendBigEndian(bytes, 28);
        for (const unsigned char value : fill) {
            bytes.push_back(255);
            bytes.insert(bytes.end(), 28 * 28 - 1, value);
        }
        return bytes;
    }

    std::vector<unsigned char> labelFile(const std::vector<unsigned char>& digits, std::uint32_t magic = 2049) {
        std::vector<unsigned char> bytes;
        appendBigEndian(bytes, magic);
        appendBigEndian(bytes, static_cast<std::uint32_t>(digits.size()));
        bytes.insert(bytes.end(), digits.begin(), digits.end());
        return bytes;
    }

    std::string writeFile(const std::string& name, const std::vector<unsigned char>& bytes) {
        const std::filesystem::path path = std::filesystem::temp_directory_path() / name;
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        return path.string();
    }

    bool loadIsRejected(const std::string& imagePath, const std::string& labelPath) {
        try {
            nnet::loadMnist(imagePath, labelPath);
        } catch (const std::runtime_error&) {
            return true;
        }
        return false;
    }
}

void runMnistTests(TestRunner& tests) {
    tests.section("MNIST LOADER");

    const std::string images = writeFile("nnet_test_images.idx", imageFile({0, 51}));
    const std::string labels = writeFile("nnet_test_labels.idx", labelFile({7, 0}));

    // A throw here is reported as a failure instead of stopping every test.
    try {
        const nnet::MnistData data = nnet::loadMnist(images, labels);
        tests.expectTrue(data.images.shape() == nnet::Tensor::Shape{2, 784},
            "images load as one flattened 784-pixel row per example");
        tests.expectTrue(data.labels.shape() == nnet::Tensor::Shape{2, 10},
            "labels load as one 10-column row per example");
        tests.expectNear(data.images.at({0, 0}), 1.0, 1.0e-12, "a pixel of 255 becomes 1");
        tests.expectNear(data.images.at({0, 1}), 0.0, 1.0e-12, "a pixel of 0 becomes 0");
        tests.expectNear(data.images.at({1, 783}), 0.2, 1.0e-12, "a pixel of 51 becomes 0.2, in the second image's last spot");
        tests.expectTrue(data.labels.getData() == std::vector<double>{0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                                                                      1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            "labels 7 and 0 become one-hot rows");
    } catch (const std::exception& error) {
        tests.expectTrue(false, std::string("well-formed test files load, but got: ") + error.what());
    }

    tests.expectTrue(loadIsRejected(writeFile("nnet_test_bad_magic.idx", imageFile({0, 51}, 2049)), labels),
        "an image file with the wrong magic number is rejected");
    // Each rejection test changes one thing from the good files, so only the
    // check under test can catch it. This one claims 27x27 but holds 28x28
    // pixels, so its size and count still match the good files exactly.
    std::vector<unsigned char> wrongSide = imageFile({0, 51});
    wrongSide[11] = 27;
    wrongSide[15] = 27;
    tests.expectTrue(loadIsRejected(writeFile("nnet_test_small_side.idx", wrongSide), labels),
        "images that are not 28x28 are rejected");
    tests.expectTrue(loadIsRejected(images, writeFile("nnet_test_three_labels.idx", labelFile({7, 0, 1}))),
        "image and label files with different counts are rejected");
    // The bad label sits in the first row, so even without the check its 1
    // would land inside the list (in the next row) rather than past its end.
    tests.expectTrue(loadIsRejected(images, writeFile("nnet_test_label_ten.idx", labelFile({10, 0}))),
        "a label above 9 is rejected");
    std::vector<unsigned char> cutShort = imageFile({0, 51});
    cutShort.pop_back();
    tests.expectTrue(loadIsRejected(writeFile("nnet_test_cut_short.idx", cutShort), labels),
        "an image file shorter than its header says is rejected");
    tests.expectTrue(loadIsRejected("nnet_test_file_that_does_not_exist.idx", labels),
        "a missing file is rejected");

    for (const char* name : {"nnet_test_images.idx", "nnet_test_labels.idx", "nnet_test_bad_magic.idx",
                             "nnet_test_small_side.idx", "nnet_test_three_labels.idx",
                             "nnet_test_label_ten.idx", "nnet_test_cut_short.idx"}) {
        std::filesystem::remove(std::filesystem::temp_directory_path() / name);
    }
}
