// npz_reader.h
//
// Minimal .npz reader. A .npz is a standard ZIP archive whose members are
// .npy files. We parse only what we need:
//   - End of Central Directory Record (EOCD), to locate central directory
//   - Central Directory File Headers, to enumerate members
//   - Local File Headers, to find each member's compressed data
//   - zlib raw inflate for deflated members (stored members copy directly)
//
// Then each member is fed to load_npy_from_memory() which understands the
// .npy v1/v2 header format.
//
// Dependencies: zlib (macOS system library; -lz). No third-party libs.
//
// Limitations (intentional, keeps it tiny):
//   - No ZIP64 (we won't have >4GB weight files)
//   - No encryption, no multi-disk, no streaming
//   - Only fp32 / fp64 / i32 / i64 dtype; extend if your model needs more

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <zlib.h>

namespace npz {

namespace fs = std::filesystem;

struct NpyArray {
    std::vector<size_t> shape;
    std::string         dtype;    // e.g. "<f4"
    std::vector<uint8_t> data;    // raw little-endian bytes (no header)

    size_t num_elements() const {
        size_t n = 1; for (size_t s : shape) n *= s; return n;
    }
    const float* as_float() const {
        return reinterpret_cast<const float*>(data.data());
    }
};

// ------------------------------------------------------------
// Parse one .npy file from a memory buffer.
// Logic mirrors loadNpy() in t2m_infer_win.cpp but reads from bytes.
// ------------------------------------------------------------
inline NpyArray load_npy_from_memory(const uint8_t* buf, size_t fsz,
                                     const std::string& name_for_err)
{
    NpyArray arr;
    if (fsz < 10 || buf[0]!=0x93 || buf[1]!='N' || buf[2]!='U' ||
        buf[3]!='M' || buf[4]!='P' || buf[5]!='Y')
        throw std::runtime_error("bad npy magic in: " + name_for_err);

    uint8_t major = buf[6];
    size_t header_len, data_start;
    if (major == 1) {
        uint16_t hl = (uint16_t)buf[8] | ((uint16_t)buf[9] << 8);
        header_len = hl;  data_start = 10 + header_len;
    } else {
        uint32_t hl = (uint32_t)buf[8]      | ((uint32_t)buf[9]  << 8)
                    | ((uint32_t)buf[10]<<16)| ((uint32_t)buf[11] << 24);
        header_len = hl;  data_start = 12 + header_len;
    }
    if (data_start > fsz)
        throw std::runtime_error("truncated npy header in: " + name_for_err);

    std::string header((const char*)&buf[data_start - header_len], header_len);

    auto find_value = [&](const std::string& key) -> std::string {
        size_t k = header.find("'" + key + "'");
        if (k == std::string::npos)
            throw std::runtime_error("npy header missing key: " + key);
        size_t colon = header.find(':', k);
        if (key == "shape") {
            size_t lp = header.find('(', colon);
            size_t rp = header.find(')', lp);
            return header.substr(lp + 1, rp - lp - 1);
        }
        size_t comma = header.find(',', colon);
        return header.substr(colon + 1, comma - colon - 1);
    };

    // descr
    std::string descr = find_value("descr");
    descr.erase(std::remove(descr.begin(), descr.end(), ' '),  descr.end());
    descr.erase(std::remove(descr.begin(), descr.end(), '\''), descr.end());
    arr.dtype = descr;

    // shape
    std::string shape_str = find_value("shape");
    size_t pos = 0;
    while (pos < shape_str.size()) {
        while (pos < shape_str.size() &&
               (shape_str[pos] == ' ' || shape_str[pos] == ','))
            pos++;
        if (pos >= shape_str.size()) break;
        size_t end = shape_str.find(',', pos);
        if (end == std::string::npos) end = shape_str.size();
        std::string num = shape_str.substr(pos, end - pos);
        if (!num.empty())
            arr.shape.push_back((size_t)std::stoull(num));
        pos = end + 1;
    }

    arr.data.assign(buf + data_start, buf + fsz);
    return arr;
}

// ------------------------------------------------------------
// Read a uint16/uint32 from a little-endian byte stream.
// ------------------------------------------------------------
inline uint16_t le_u16(const uint8_t* p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}
inline uint32_t le_u32(const uint8_t* p) {
    return  (uint32_t)p[0]        | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

// ------------------------------------------------------------
// Decompress a deflated member with raw inflate (no zlib wrapper).
// ------------------------------------------------------------
inline std::vector<uint8_t> raw_inflate(const uint8_t* src, size_t src_len,
                                        size_t uncompressed_len,
                                        const std::string& name_for_err)
{
    std::vector<uint8_t> out(uncompressed_len);
    z_stream zs{};
    zs.next_in   = const_cast<Bytef*>(src);
    zs.avail_in  = (uInt)src_len;
    zs.next_out  = out.data();
    zs.avail_out = (uInt)uncompressed_len;

    // -MAX_WBITS = raw deflate (no zlib/gzip header), which is what ZIP uses
    if (inflateInit2(&zs, -MAX_WBITS) != Z_OK)
        throw std::runtime_error("inflateInit2 failed for: " + name_for_err);
    int rc = inflate(&zs, Z_FINISH);
    inflateEnd(&zs);
    if (rc != Z_STREAM_END)
        throw std::runtime_error("inflate failed for: " + name_for_err +
                                 " (rc=" + std::to_string(rc) + ")");
    return out;
}

// ------------------------------------------------------------
// Load a .npz file and return a name -> NpyArray map.
//
// We do a single-pass: read whole file into memory (these are 3-4 MB
// in our case, trivially small), parse EOCD, walk central directory.
// ------------------------------------------------------------
inline std::unordered_map<std::string, NpyArray> load_npz(const fs::path& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open npz: " + path.string());
    f.seekg(0, std::ios::end);
    size_t fsz = (size_t)f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf(fsz);
    f.read((char*)buf.data(), fsz);

    // ---- find EOCD: signature 0x06054b50, near the end ----
    // EOCD is 22 bytes + comment (typically 0). Search backwards from end.
    const uint32_t EOCD_SIG = 0x06054b50;
    const size_t   EOCD_MIN = 22;
    if (fsz < EOCD_MIN) throw std::runtime_error("npz too small");

    ssize_t eocd_off = -1;
    size_t  search_from = fsz - EOCD_MIN;
    size_t  search_to   = (fsz > 0xFFFF + EOCD_MIN) ? (fsz - 0xFFFF - EOCD_MIN) : 0;
    for (ssize_t i = (ssize_t)search_from; i >= (ssize_t)search_to; --i) {
        if (le_u32(&buf[i]) == EOCD_SIG) { eocd_off = i; break; }
    }
    if (eocd_off < 0) throw std::runtime_error("EOCD not found (corrupt npz?)");

    uint16_t n_entries = le_u16(&buf[eocd_off + 10]);  // total entries on this disk
    uint32_t cd_size   = le_u32(&buf[eocd_off + 12]);
    uint32_t cd_off    = le_u32(&buf[eocd_off + 16]);
    (void)cd_size;

    // ---- walk central directory ----
    const uint32_t CD_SIG  = 0x02014b50;
    const uint32_t LFH_SIG = 0x04034b50;

    std::unordered_map<std::string, NpyArray> out;
    size_t p = cd_off;
    for (uint16_t e = 0; e < n_entries; ++e) {
        if (p + 46 > fsz || le_u32(&buf[p]) != CD_SIG)
            throw std::runtime_error("bad central dir entry at " +
                                     std::to_string(p));
        uint16_t method     = le_u16(&buf[p + 10]);
        uint32_t comp_sz    = le_u32(&buf[p + 20]);
        uint32_t uncomp_sz  = le_u32(&buf[p + 24]);
        uint16_t name_len   = le_u16(&buf[p + 28]);
        uint16_t extra_len  = le_u16(&buf[p + 30]);
        uint16_t cmt_len    = le_u16(&buf[p + 32]);
        uint32_t lfh_off    = le_u32(&buf[p + 42]);

        std::string name((const char*)&buf[p + 46], name_len);
        p += 46 + name_len + extra_len + cmt_len;

        // ---- jump to local file header to find data offset ----
        if (lfh_off + 30 > fsz || le_u32(&buf[lfh_off]) != LFH_SIG)
            throw std::runtime_error("bad local file header for: " + name);
        uint16_t lfh_name_len  = le_u16(&buf[lfh_off + 26]);
        uint16_t lfh_extra_len = le_u16(&buf[lfh_off + 28]);
        size_t   data_off      = lfh_off + 30 + lfh_name_len + lfh_extra_len;
        if (data_off + comp_sz > fsz)
            throw std::runtime_error("data out of bounds for: " + name);

        // ---- extract bytes ----
        std::vector<uint8_t> npy_bytes;
        if (method == 0) {           // stored
            npy_bytes.assign(&buf[data_off], &buf[data_off] + comp_sz);
        } else if (method == 8) {    // deflated
            npy_bytes = raw_inflate(&buf[data_off], comp_sz, uncomp_sz, name);
        } else {
            throw std::runtime_error("unsupported zip method " +
                                     std::to_string(method) +
                                     " for member: " + name);
        }

        // ---- parse the .npy bytes ----
        // numpy savez writes members as "<name>.npy"; strip that for the key
        std::string key = name;
        if (key.size() > 4 && key.substr(key.size() - 4) == ".npy")
            key = key.substr(0, key.size() - 4);

        out.emplace(std::move(key),
                    load_npy_from_memory(npy_bytes.data(), npy_bytes.size(), name));
    }

    return out;
}

} // namespace npz
