#ifndef NBIT_ARRAY_H_
#define NBIT_ARRAY_H_

//namespace nbit_array {

template <typename ItemType>
class UIntArray {
    size_t byteCount;
    ItemType* data;
public:
    UIntArray(size_t size) {
        byteCount = sizeof(ItemType[size]);
        data = new ItemType[size]();
    }
    ~UIntArray() {
        delete[] data;
    }
    inline ItemType get(size_t index) {
        return data[index];
    }
    inline void set(size_t index, ItemType value) {
        data[index] = value;
    }
    inline ItemType mask(ItemType fingerprint) {
        return fingerprint;
    }
    size_t getByteCount() {
        return byteCount;
    }
};

template <typename ItemType, size_t bitsPerEntry, uint32_t bitMask = (1 << bitsPerEntry) - 1>
class NBitArray {
    size_t byteCount;
    uint8_t* data;
public:
    NBitArray(size_t size) {
        byteCount = (size * bitsPerEntry + 63 + 128) / 64 * 64 / 8;
        data = new uint8_t[byteCount]();
    }
    ~NBitArray() {
        delete[] data;
    }
    inline ItemType get(size_t index) {
        size_t bitPos = index * bitsPerEntry;
        size_t firstBytePos = (size_t) (bitPos >> 3);

        uint32_t word = __builtin_bswap32(*((uint32_t*) (data + firstBytePos))) >> 8;
/*
        uint32_t word = ((data[firstBytePos] & 0xff) << 16) |
                ((data[firstBytePos + 1] & 0xff) << 8) |
                ((data[firstBytePos + 2] & 0xff));
*/
        return (ItemType) ((word >> (24 - bitsPerEntry - (bitPos & 7))) & bitMask);

    }
    inline void set(size_t index, ItemType value) {
        size_t bitPos = index * bitsPerEntry;
        size_t firstBytePos = (size_t) (bitPos >> 3);

        uint32_t word = __builtin_bswap32(*((uint32_t*) (data + firstBytePos))) >> 8;
/*
        uint32_t word = ((data[firstBytePos] & 0xff) << 16) |
                ((data[firstBytePos + 1] & 0xff) << 8) |
                ((data[firstBytePos + 2] & 0xff));
*/
        word &= ~(bitMask << (24 - bitsPerEntry - (bitPos & 7)));
        word |= ((value & bitMask) << (24 - bitsPerEntry - (bitPos & 7)));

        data[firstBytePos] = (uint8_t) (word >> 16);
        data[firstBytePos + 1] = (uint8_t) (word >> 8);
        data[firstBytePos + 2] = (uint8_t) word;
    }
    inline ItemType mask(ItemType fingerprint) {
        return fingerprint & bitMask;
    }
    size_t getByteCount() {
        return byteCount;
    }
};


// }  // namespace n_bit_array

#endif  // NBIT_ARRAY_H_
