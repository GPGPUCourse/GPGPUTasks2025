#pragma once

#include <memory>
#include <cstddef>
#include <cassert>
#include <string>
#include <vector>
#include <memory>

#include <libbase/data_type.h>

namespace cimg_library {
    template<typename T>
    class CImg;
}

using cimg_library::CImg;

template <typename T>
class CImgWrapper {
public:
    std::shared_ptr<CImg<T>> img;

    CImgWrapper(CImg<T>& img) : img(std::make_shared<CImg<T>>(img)) {}
    CImgWrapper(size_t width, size_t height, size_t cn) : img(std::make_shared<CImg<T>>(width, height, 1, cn)) {}
};

typedef int mouse_click_t;

const mouse_click_t MOUSE_LEFT = 1;
const mouse_click_t MOUSE_RIGHT = 2;
const mouse_click_t MOUSE_MIDDLE = 4;

typedef unsigned int keycode_t;
keycode_t getKeyCode(const char* keycode);
inline keycode_t getEscapeKeyCode() { return getKeyCode("ESC"); }

class CImgDisplayWrapper;

class AnyImage;

template <typename T>
class TypedImage;

class ImageWindow {
public:
    ImageWindow(std::string title);

    void display(const AnyImage &image);
    template <typename T>
    void display(const TypedImage<T> &image);
    void resize(size_t width, size_t height);
    void resize();
    void setTitle(std::string title);

    bool isClosed();
    bool isResized();

    keycode_t wait(unsigned int milliseconds);

    mouse_click_t getMouseClick();
    int getMouseX();
    int getMouseY();

    size_t width();
    size_t height();

protected:
    std::shared_ptr<CImgDisplayWrapper> cimg_display;
    std::string title;
};

// wait all windows until Escape is pressed or all windows were closed
void waitAllWindows(std::vector<ImageWindow> windows);

bool isAnyWindowClosed(std::vector<ImageWindow> windows);
bool isEscapePressed(std::vector<ImageWindow> windows, unsigned int milliseconds=10);

class AnyImage {
public:
    AnyImage();
    AnyImage(size_t width, size_t height, size_t cn, DataType type);
    AnyImage(const AnyImage &image);
    AnyImage(const AnyImage &image, size_t offset_x, size_t offset_y, size_t width, size_t height);

    size_t      width() const       { return width_;                }
    size_t      height() const      { return height_;               }
    size_t      channels() const    { return cn_;                   }
    DataType    type() const        { return type_;                 }

    bool        isNull()            { return data_ == nullptr;      }
    void*       ptr()               { return data_;                 }
    const void* ptr() const         { return data_;                 }
    void*       ptr(size_t j)       { rassert(j < height_, 359644174); return data_ + j * stride_ * dataSize(type_); }
    const void* ptr(size_t j) const { rassert(j < height_, 976926074); return data_ + j * stride_ * dataSize(type_); }
    void*       ptr(size_t j, size_t i)       { rassert(i < width_, 1346134613631); return (unsigned char*)       ptr(j) + i * cn_ * dataSize(type_); }
    const void* ptr(size_t j, size_t i) const { rassert(i < width_, 2463463465343); return (const unsigned char*) ptr(j) + i * cn_ * dataSize(type_); }

    size_t      stride() const      { return stride_;               } // measured in elements, not in bytes (i.e. mostly equal to width*cn)

    ImageWindow show() const;
    ImageWindow show(const char* title) const;
	template <typename T>
	CImgWrapper<T> toCImg() const {
		TypedImage<T> img(*this);
		return img.toCImg();
	}

    void saveJPEG(const char *const filename, int quality=100) const;
    void saveJPEG(const std::string& filename, int quality=100) const;
    void savePNG(const char *const filename) const;
    void savePNG(const std::string& filename) const;
    void saveBMP(const char *const filename) const;
    void saveBMP(const std::string& filename) const;

protected:
    void        allocateData();
    void        init();
    void        init(size_t width, size_t height, size_t cn, DataType type);
    void        init(const AnyImage &image);

    size_t      width_;
    size_t      height_;
    size_t      cn_;
    DataType    type_;

    size_t      stride_;

    std::shared_ptr<unsigned char> data_buffer_;
    unsigned char                  *data_;
};

template <typename T>
class TypedImage : public AnyImage {
public:
    TypedImage();
    TypedImage(size_t width, size_t height, size_t cn);
    TypedImage(const TypedImage<T> &image);
	TypedImage(const AnyImage &image);
	TypedImage(const TypedImage<T> &image, size_t offset_x, size_t offset_y, size_t width, size_t height);

    TypedImage(const char *const filename);
    TypedImage(const std::string& filename);

    void fromCImg(CImgWrapper<T>& wrapper);
    CImgWrapper<T> toCImg() const;

    TypedImage copy() const;
    TypedImage<T>& operator=(const TypedImage<T>& that) = default;

    void fill(T value);
    void fill(T value[]);
    void fillZero() { fill((T) 0); }
    void replace(T a, T b);
    void replace(T a[], T b[]);

    T*          ptr()               { return (T*)       AnyImage::ptr();    }
    const T*    ptr() const         { return (const T*) AnyImage::ptr();    }
    T*          ptr(size_t j)       { return (T*)       AnyImage::ptr(j);   }
    const T*    ptr(size_t j) const { return (const T*) AnyImage::ptr(j);   }

    T& operator()(size_t row, size_t col) {
        rassert(col < width_, col, width_, 514336953);
        return ptr(row)[col * cn_];
    }

    T& operator()(size_t row, size_t col, size_t c) {
        rassert(col < width_, col, width_, 314336923553);
        rassert(c < cn_, c, cn_, 843906761);
        return ptr(row)[col * cn_ + c];
    }

    T operator()(size_t row, size_t col) const {
        rassert(col < width_, col, width_, 51433695353);
        return ptr(row)[col * cn_];
    }

    T operator()(size_t row, size_t col, size_t c) const {
        rassert(col < width_, col, width_, 514336923553);
        rassert(c < cn_, c, cn_, 2143906761);
        return ptr(row)[col * cn_ + c];
    }

    double sample(double x, double y, size_t c) const;
    T sample_casted(double x, double y, size_t c) const;

protected:
};

typedef TypedImage<unsigned char>   image8u;
typedef TypedImage<unsigned int>    image32u;
typedef TypedImage<int>             image32i;
typedef TypedImage<float>           image32f;
