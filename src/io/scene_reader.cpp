#include "scene_reader.h"

#include <fstream>
#include <sstream>
#include <cstdint>
#include <algorithm>
#include <cctype>
#include <string>
#include <vector>
#include <cstdlib> // strtof

#include "libutils/misc.h" // rassert, ends_with

// ----------------- Helpers (OBJ) -----------------

// Fast ASCII-only whitespace check (faster than std::isspace with locale)
static inline bool isSpaceChar(char c) {
    return c == ' '  || c == '\t' ||
        c == '\r' || c == '\n' ||
        c == '\f' || c == '\v';
}

// Fast decimal float parser: [sign] digits [ '.' digits ]
// Does NOT handle exponent; caller must ensure there is no 'e'/'E' in [s,end).
static inline float fastAtofSimple(const char* s, const char* end) {
    bool neg = false;
    if (s < end && (*s == '+' || *s == '-')) {
        neg = (*s == '-');
        ++s;
    }

    double intPart = 0.0;
    while (s < end && *s >= '0' && *s <= '9') {
        intPart = intPart * 10.0 + double(*s - '0');
        ++s;
    }

    double fracPart = 0.0;
    if (s < end && *s == '.') {
        ++s;
        double base = 0.1;
        while (s < end && *s >= '0' && *s <= '9') {
            fracPart += double(*s - '0') * base;
            base *= 0.1;
            ++s;
        }
    }

    double res = intPart + fracPart;
    return neg ? float(-res) : float(res);
}

// Parse leading integer index from token sequence like "12/3/4" or "-2//5"
// Works on a char* pointer and advances it to the end of token.
// Returns 0 only when there are no more tokens on the line.
static inline long parseObjIndexToken(const char *&p) {
    // skip leading spaces
    while (*p && isSpaceChar(*p))
        ++p;

    // no more tokens on this line
    if (!*p)
        return 0;

    bool neg = false;
    if (*p == '-') {
        neg = true;
        ++p;
    }

    long v = 0;
    const char* digitStart = p;

    // read integer digits
    while (*p >= '0' && *p <= '9') {
        v = v * 10 + (*p - '0');
        ++p;
    }

    // must have at least one digit
    rassert(p != digitStart, 1004, "invalid OBJ index token");

    // skip the rest of token (/, vt/vn, etc.) until whitespace
    while (*p && !isSpaceChar(*p))
        ++p;

    return neg ? -v : v;
}

// Parse float from char* and advance pointer
static inline float parseFloat(const char *&p) {
    // skip spaces
    while (*p && isSpaceChar(*p))
        ++p;

    rassert(*p, 1202); // no data

    const char* start = p;
    const char* q = start;
    bool needSlow = false;

    // find token end and detect exponent / special values
    while (*q && !isSpaceChar(*q)) {
        char c = *q;
        // exponent or nan/inf -> use slow path
        if (c == 'e' || c == 'E' ||
            c == 'n' || c == 'N' ||
            c == 'i' || c == 'I') {
            needSlow = true;
        }
        ++q;
    }

    float v = 0.0f;

    if (!needSlow) {
        // fast simple decimal
        v = fastAtofSimple(start, q);
    } else {
        // fallback to libc parser
        char *end = nullptr;
        v = std::strtof(start, &end);
        rassert(end != start, 1203); // parse failed
    }

    // move pointer to end of token
    p = q;
    return v;
}

// Convert OBJ index (1-based or negative) to 0-based
static inline uint32_t resolveObjIndex(long idx, size_t vcount) {
    if (idx > 0) {
        rassert(static_cast<size_t>(idx) <= vcount, 1001);
        return static_cast<uint32_t>(idx - 1);
    } else if (idx < 0) {
        long pos = static_cast<long>(vcount) + idx; // -1 == last
        rassert(pos >= 0 && pos < static_cast<long>(vcount), 1002);
        return static_cast<uint32_t>(pos);
    }
    rassert(false, 1003); // zero is invalid in OBJ
    return 0;
}

// Fan triangulation: (v0, vi, vi+1)
template <class Idx>
static inline void triangulateFan(const std::vector<Idx>& poly, std::vector<point3u>& out) {
    if (poly.size() < 3) return;
    for (size_t i = 1; i + 1 < poly.size(); ++i) {
        out.emplace_back(point3u{
            static_cast<uint32_t>(poly[0]),
            static_cast<uint32_t>(poly[i]),
            static_cast<uint32_t>(poly[i+1])
        });
    }
}

// ----------------- OBJ loader -----------------

SceneGeometry loadOBJ(const std::string &path)
{
    std::ifstream in(path, std::ios::binary);
    rassert(in.good(), 1100, path);

    // use larger I/O buffer to reduce number of syscalls and getline overhead
    std::vector<char> ioBuffer(1 << 20); // 1 MiB
    in.rdbuf()->pubsetbuf(ioBuffer.data(), ioBuffer.size());

    SceneGeometry scene;
    scene.vertices.reserve(1 << 12);
    scene.faces.reserve(1 << 12);

    std::string line;

    // reuse polygon buffer for all faces to avoid reallocations
    std::vector<uint32_t> poly;
    poly.reserve(8);

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') continue;

        // vertex: "v x y z"
        if (line.size() > 2 && line[0] == 'v' && isSpaceChar(line[1])) {
            const char *p = line.c_str() + 2;
            float x = parseFloat(p);
            float y = parseFloat(p);
            float z = parseFloat(p);
            scene.vertices.emplace_back(point3f{x,y,z});
            continue;
        }

        // face: "f a b c ..." with tokens a[/b[/c]]
        if (line.size() > 2 && line[0] == 'f' && isSpaceChar(line[1])) {
            const char *p = line.c_str() + 2;
            poly.clear();

            while (*p) {
                long raw = parseObjIndexToken(p);
                if (raw == 0) break; // no more indices
                uint32_t idx = resolveObjIndex(raw, scene.vertices.size());
                poly.push_back(idx);
            }

            rassert(poly.size() >= 3, 1301, path);
            triangulateFan(poly, scene.faces);
            continue;
        }

        // other records are ignored (vt, vn, usemtl, mtllib, g, o, s, ...)
    }

    rassert(!scene.vertices.empty(), 1901, path);
    rassert(!scene.faces.empty(),    1902, path);
    return scene;
}

// ----------------- Helpers (PLY) -----------------

namespace ply_detail {

// Map PLY scalar type to byte size
static inline size_t typeSize(const std::string& t) {
    if (t=="char"||t=="int8")       return 1;
    if (t=="uchar"||t=="uint8")     return 1;
    if (t=="short"||t=="int16")     return 2;
    if (t=="ushort"||t=="uint16")   return 2;
    if (t=="int"||t=="int32")       return 4;
    if (t=="uint"||t=="uint32")     return 4;
    if (t=="float"||t=="float32")   return 4;
    if (t=="double"||t=="float64")  return 8;
    rassert(false, 2101, t);
    return 0;
}

// Read little-endian value of type T and fix endianness on big-endian hosts
template <class T>
static inline T readLE(std::istream& is) {
    T v{};
    is.read(reinterpret_cast<char*>(&v), sizeof(T));
    rassert(is.good(), 2201);
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
    std::reverse(reinterpret_cast<unsigned char*>(&v),
                 reinterpret_cast<unsigned char*>(&v) + sizeof(T));
#endif
    return v;
}

static inline uint64_t readScalarAsU64(std::istream& is, const std::string& t) {
    if (t=="uchar"||t=="uint8")   return readLE<uint8_t>(is);
    if (t=="char" ||t=="int8")    return static_cast<uint64_t>(readLE<int8_t>(is));
    if (t=="ushort"||t=="uint16") return readLE<uint16_t>(is);
    if (t=="short"||t=="int16")   return static_cast<uint64_t>(readLE<int16_t>(is));
    if (t=="uint"||t=="uint32")   return readLE<uint32_t>(is);
    if (t=="int" ||t=="int32")    return static_cast<uint64_t>(readLE<int32_t>(is));
    rassert(false, 2301, t);
    return 0;
}

static inline uint32_t readIndexAsU32(std::istream& is, const std::string& t) {
    if (t=="uchar"||t=="uint8")   return readLE<uint8_t>(is);
    if (t=="char" ||t=="int8")    return static_cast<uint32_t>(readLE<int8_t>(is));
    if (t=="ushort"||t=="uint16") return readLE<uint16_t>(is);
    if (t=="short"||t=="int16")   return static_cast<uint32_t>(readLE<int16_t>(is));
    if (t=="uint"||t=="uint32")   return readLE<uint32_t>(is);
    if (t=="int" ||t=="int32")    return static_cast<uint32_t>(readLE<int32_t>(is));
    rassert(false, 2302, t);
    return 0;
}

static inline double readNumberAsDouble(std::istream& is, const std::string& t) {
    if (t=="float"||t=="float32")   return static_cast<double>(readLE<float>(is));
    if (t=="double"||t=="float64")  return readLE<double>(is);
    if (t=="char"||t=="int8")       return static_cast<double>(readLE<int8_t>(is));
    if (t=="uchar"||t=="uint8")     return static_cast<double>(readLE<uint8_t>(is));
    if (t=="short"||t=="int16")     return static_cast<double>(readLE<int16_t>(is));
    if (t=="ushort"||t=="uint16")   return static_cast<double>(readLE<uint16_t>(is));
    if (t=="int"||t=="int32")       return static_cast<double>(readLE<int32_t>(is));
    if (t=="uint"||t=="uint32")     return static_cast<double>(readLE<uint32_t>(is));
    rassert(false, 2303, t);
    return 0.0;
}

} // namespace ply_detail

// ----------------- PLY loader -----------------

SceneGeometry loadPLY(const std::string &path)
{
    using namespace ply_detail;

    std::ifstream in(path);
    rassert(in.good(), 3100, path);

    // Header
    std::string line;
    std::getline(in, line);
    rassert(line == "ply", 3101, path);

    bool isAscii = false;
    bool isBinaryLE = false;

    uint64_t vertexCount = 0;
    uint64_t faceCount   = 0;

    struct VProp { std::string name, type; };
    std::vector<VProp> vprops;
    int xPos = -1, yPos = -1, zPos = -1;

    struct FaceProp {
        bool isList;
        std::string countType, itemType, name, type;
    };
    std::vector<FaceProp> fprops;

    enum Section { NONE, VERTEX, FACE, OTHER };
    Section section = NONE;

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line == "end_header") break;

        std::istringstream ss(line);
        std::string tok;
        ss >> tok;

        if (tok == "format") {
            std::string fmt; ss >> fmt;
            rassert(!fmt.empty(), 3102, path);
            if (fmt == "ascii") isAscii = true;
            else if (fmt == "binary_little_endian") isBinaryLE = true;
            else rassert(false, 3103, fmt);
            continue;
        }

        if (tok == "comment") continue;

        if (tok == "element") {
            std::string name; uint64_t cnt=0;
            ss >> name >> cnt;
            rassert(!ss.fail(), 3104, path);

            if (name == "vertex") {
                vertexCount = cnt; section = VERTEX; vprops.clear();
            } else if (name == "face") {
                faceCount = cnt; section = FACE; fprops.clear();
            } else {
                section = OTHER;
            }
            continue;
        }

        if (tok == "property") {
            if (section == VERTEX) {
                std::string t, name;
                ss >> t;
                rassert(!ss.fail(), 3105);
                rassert(t != "list", 3105, "list property in vertex not supported");
                ss >> name;
                rassert(!ss.fail(), 3106);
                vprops.push_back({name, t});
                if (name=="x") xPos = int(vprops.size())-1;
                else if (name=="y") yPos = int(vprops.size())-1;
                else if (name=="z") zPos = int(vprops.size())-1;
            } else if (section == FACE) {
                std::string t; ss >> t;
                if (t == "list") {
                    std::string ct, it, name;
                    ss >> ct >> it >> name;
                    rassert(!ss.fail(), 3107);
                    fprops.push_back({true, ct, it, name, ""});
                } else {
                    std::string name; ss >> name;
                    rassert(!ss.fail(), 3108);
                    fprops.push_back({false, "", "", name, t});
                }
            } else {
                // ignore properties of other elements
            }
            continue;
        }
        // ignore unknown header tokens
    }

    rassert((isAscii || isBinaryLE) && !(isAscii && isBinaryLE), 3109);
    rassert(vertexCount > 0, 3110);
    rassert(faceCount > 0,   3111);
    rassert(xPos >= 0 && yPos >= 0 && zPos >= 0, 3112, "x/y/z missing in vertex");

    SceneGeometry scene;
    scene.vertices.resize(static_cast<size_t>(vertexCount));
    scene.faces.reserve(static_cast<size_t>(faceCount));

    if (isAscii) {
        // ASCII vertices
        for (uint64_t i = 0; i < vertexCount; ++i) {
            std::string vline;
            std::getline(in, vline);
            rassert(!in.fail(), 3200);
            std::istringstream vs(vline);
            std::vector<double> scalars; scalars.reserve(vprops.size());
            for (size_t p = 0; p < vprops.size(); ++p) {
                double val = 0.0; vs >> val; rassert(!vs.fail(), 3201);
                scalars.push_back(val);
            }
            scene.vertices[static_cast<size_t>(i)] =
                point3f{ float(scalars[size_t(xPos)]),
                         float(scalars[size_t(yPos)]),
                         float(scalars[size_t(zPos)]) };
        }

        // Identify face indices list
        int faceListIdx = -1;
        for (size_t i = 0; i < fprops.size(); ++i)
            if (fprops[i].isList && (fprops[i].name=="vertex_indices" || fprops[i].name=="vertex_index"))
                faceListIdx = int(i);
        rassert(faceListIdx >= 0, 3202, "no vertex_indices list");

        // ASCII faces
        for (uint64_t i = 0; i < faceCount; ++i) {
            std::string fline;
            std::getline(in, fline);
            rassert(!in.fail(), 3203);
            std::istringstream fs(fline);

            std::vector<uint32_t> polygon;
            for (size_t p = 0; p < fprops.size(); ++p) {
                if (fprops[p].isList) {
                    uint64_t k = 0; fs >> k; rassert(!fs.fail(), 3204);
                    std::vector<uint32_t> tmp; tmp.resize(static_cast<size_t>(k));
                    for (size_t j = 0; j < k; ++j) { uint64_t id; fs >> id; rassert(!fs.fail(), 3205); tmp[j] = static_cast<uint32_t>(id); }
                    if ((int)p == faceListIdx) polygon = std::move(tmp);
                } else {
                    double dummy = 0.0; fs >> dummy; rassert(!fs.fail(), 3206);
                }
            }
            rassert(polygon.size() >= 3, 3207);
            triangulateFan(polygon, scene.faces);
        }
    } else {
        // Binary little-endian

        // Binary vertices: read each declared property in order
        for (uint64_t i = 0; i < vertexCount; ++i) {
            std::vector<double> vals; vals.reserve(vprops.size());
            for (size_t p = 0; p < vprops.size(); ++p) {
                vals.push_back(readNumberAsDouble(in, vprops[p].type));
            }
            rassert(in.good(), 3300);
            scene.vertices[static_cast<size_t>(i)] =
                point3f{ float(vals[size_t(xPos)]),
                         float(vals[size_t(yPos)]),
                         float(vals[size_t(zPos)]) };
        }

        // Binary faces: consume properties in declared order
        int faceListIdx = -1;
        for (size_t i = 0; i < fprops.size(); ++i)
            if (fprops[i].isList && (fprops[i].name=="vertex_indices" || fprops[i].name=="vertex_index"))
                faceListIdx = int(i);
        rassert(faceListIdx >= 0, 3302, "no vertex_indices list");

        for (uint64_t i = 0; i < faceCount; ++i) {
            std::vector<uint32_t> polygon;

            for (size_t p = 0; p < fprops.size(); ++p) {
                const auto& fp = fprops[p];
                if (fp.isList) {
                    uint64_t k = readScalarAsU64(in, fp.countType);
                    rassert(k >= 3 && k < (1ull<<20), 3303);
                    std::vector<uint32_t> tmp; tmp.reserve(static_cast<size_t>(k));
                    for (uint64_t j = 0; j < k; ++j) tmp.push_back(readIndexAsU32(in, fp.itemType));
                    if ((int)p == faceListIdx) polygon = std::move(tmp);
                } else {
                    // discard scalar face property
                    (void)readNumberAsDouble(in, fp.type);
                }
            }

            rassert(!polygon.empty(), 3304);
            triangulateFan(polygon, scene.faces);
        }
    }

    rassert(!scene.vertices.empty(), 3900, path);
    rassert(!scene.faces.empty(),    3901, path);
    return scene;
}

// ----------------- Dispatcher -----------------

SceneGeometry loadScene(const std::string &path)
{
    if (ends_with(path, ".ply")) {
        return loadPLY(path);
    } else if (ends_with(path, ".obj")) {
        return loadOBJ(path);
    } else {
        rassert(false, 324134123142132, path);
    }
}
