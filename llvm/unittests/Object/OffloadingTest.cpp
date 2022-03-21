#include "llvm/Object/OffloadBinary.h"

#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <random>

TEST(OffloadingTest, checkOffloadingBinary) {
  // Create random data to fill the image.
  std::mt19937 Rng(std::random_device{}());
  std::uniform_int_distribution<uint64_t> SizeDist(0, 256);
  std::uniform_int_distribution<uint16_t> KindDist(0);
  std::uniform_int_distribution<uint8_t> BinaryDist(0);
  std::uniform_int_distribution<char> StringDist('!', '~');
  std::vector<uint8_t> Image(SizeDist(Rng));
  std::generate(Image.begin(), Image.end(), [&]() { return BinaryDist(Rng); });
  std::vector<std::pair<std::string, std::string>> Strings(SizeDist(Rng));
  for (auto &KeyAndValue : Strings) {
    std::string Key(SizeDist(Rng), '\0');
    std::string Value(SizeDist(Rng), '\0');

    std::generate(Key.begin(), Key.end(), [&]() { return StringDist(Rng); });
    std::generate(Value.begin(), Value.end(),
                  [&]() { return StringDist(Rng); });

    KeyAndValue = std::make_pair(Key, Value);
  }

  // Create the image.
  llvm::StringMap<llvm::StringRef> StringData;
  for (auto &KeyAndValue : Strings)
    StringData[KeyAndValue.first] = KeyAndValue.second;
  std::unique_ptr<llvm::MemoryBuffer> ImageData =
      llvm::MemoryBuffer::getMemBuffer(
          {reinterpret_cast<char *>(Image.data()), Image.size()}, "", false);

  llvm::OffloadBinary::OffloadingImage Data;
  Data.TheImageKind = static_cast<llvm::ImageKind>(KindDist(Rng));
  Data.TheOffloadKind = static_cast<llvm::OffloadKind>(KindDist(Rng));
  Data.Flags = KindDist(Rng);
  Data.StringData = StringData;
  Data.Image = *ImageData;

  auto BinaryBuffer = llvm::OffloadBinary::write(Data);

  auto BinaryOrErr = llvm::OffloadBinary::read(*BinaryBuffer);
  if (!BinaryOrErr)
    FAIL();

  // Make sure we get the same data out.
  auto &Binary = **BinaryOrErr;
  ASSERT_EQ(Data.TheImageKind,
            static_cast<llvm::ImageKind>(Binary.getImageKind()));
  ASSERT_EQ(Data.TheOffloadKind,
            static_cast<llvm::OffloadKind>(Binary.getOffloadKind()));
  ASSERT_EQ(Data.Flags, Binary.getFlags());

  for (auto &KeyAndValue : Strings)
    ASSERT_TRUE(StringData[KeyAndValue.first] ==
                Binary.getString(KeyAndValue.first));

  EXPECT_TRUE(Data.Image.getBuffer() == Binary.getImage());

  // Ensure the size and alignment of the data is correct.
  EXPECT_TRUE(Binary.getSize() % llvm::OffloadBinary::getAlignment() == 0);
  EXPECT_TRUE(Binary.getSize() == BinaryBuffer->getBuffer().size());
}
