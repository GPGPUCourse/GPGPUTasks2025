#include "images.h"

#include <cassert>

//#define cimg_use_jpeg

// Configuring display system in CImg (See http://cimg.eu/reference/group__cimg__environment.html):
#ifdef __unix__
    // X11-based
    #define cimg_display 1
#elif defined _WIN32
    #define cimg_display 2
#else
    #define cimg_display 0
#endif
// CImg header-only library available at https://github.com/dtschump/CImg (v.2.3.4 0f0d65d984b08ad8178969f4fa4d1641d721354b)
#include "CImg.h"
#undef cimg_display

using namespace cimg_library;

AnyImage::AnyImage() {
	init();
}

AnyImage::AnyImage(size_t width, size_t height, size_t cn, DataType type) {
	init(width, height, cn, type);

	allocateData();
}

AnyImage::AnyImage(const AnyImage &image) {
	init(image.width(), image.height(), image.channels(), image.type());

	stride_         = image.stride_;
	data_buffer_    = image.data_buffer_;
	data_           = image.data_;
}

AnyImage::AnyImage(const AnyImage &image, size_t offset_x, size_t offset_y, size_t width, size_t height) {
	init(width, height, image.channels(), image.type());

	rassert(offset_x + width <= image.width(), offset_x, width, image.width(), 507985238);
	rassert(offset_y + height <= image.height(), offset_y, height, image.height(), 324513412);
	stride_           = image.stride_;
	data_buffer_    = image.data_buffer_;
	data_           = image.data_ + (offset_y * image.stride() + offset_x * image.channels()) * dataSize(image.type());
}

void AnyImage::allocateData() {
	data_buffer_ = std::shared_ptr<unsigned char>(new unsigned char[width_ * height_ * cn_ * dataSize(type_)]);
	data_ = data_buffer_.get();
}

void AnyImage::init() {
	init(0, 0, 0, DataTypeUndefined);
}

void AnyImage::init(size_t width, size_t height, size_t cn, DataType type) {
	width_          = width;
	height_         = height;
	cn_             = cn;
	type_           = type;

	stride_           = width_ * cn_;
	data_buffer_.reset();
	data_           = nullptr;
}

void AnyImage::init(const AnyImage &image) {
	width_          = image.width_;
	height_         = image.height_;
	cn_             = image.cn_;
	type_           = image.type_;
	stride_         = image.stride_;
	data_buffer_    = image.data_buffer_;
	data_           = image.data_;
}

ImageWindow AnyImage::show() const {
	return show("");
}

ImageWindow AnyImage::show(const char *title) const {
	std::string full_title = std::string("CImg ") + title;
	ImageWindow window = ImageWindow(full_title);
	window.display(*this);

	// workaround for i3wm - it opens window in fullscreen by default,
	// you can open window in floating mod by adding to the end of ~/.config/i3/config
	// for_window [title="^CImg.*] floating enable
	window.wait(10);
	window.resize(width(), height());

	return window;
}

void AnyImage::saveJPEG(const char *const filename, int quality) const {
	if (type() == DataType8u) this->toCImg<unsigned char>().img->save_jpeg(filename, quality);
	else throwUnsupportedDataType(type());
}

void AnyImage::saveJPEG(const std::string& filename, int quality) const {
	saveJPEG(filename.c_str(), quality);
}

void AnyImage::savePNG(const char *const filename) const {
	if (type() == DataType8u) this->toCImg<unsigned char>().img->save_png(filename);
	else throwUnsupportedDataType(type());
}

void AnyImage::savePNG(const std::string& filename) const {
	savePNG(filename.c_str());
}

void AnyImage::saveBMP(const char *const filename) const {
	if (type() == DataType8u) this->toCImg<unsigned char>().img->save_bmp(filename);
	else throwUnsupportedDataType(type());
}

void AnyImage::saveBMP(const std::string& filename) const {
        saveBMP(filename.c_str());
}

template <typename T>
TypedImage<T>::TypedImage() : AnyImage() {
	type_ = DataTypeTraits<T>::type();
}

template <typename T>
TypedImage<T>::TypedImage(size_t width, size_t height, size_t cn) : AnyImage(width, height, cn, DataTypeTraits<T>::type())
{}

template <typename T>
TypedImage<T>::TypedImage(const TypedImage<T> &that) : AnyImage(that) {}

template <typename T>
TypedImage<T>::TypedImage(const AnyImage &image) : AnyImage(image) {
	rassert(image.type() == DataTypeTraits<T>::type(), image.type(), DataTypeTraits<T>::type(), 711430549);
}

template <typename T>
TypedImage<T>::TypedImage(const TypedImage<T> &image, size_t offset_x, size_t offset_y, size_t width, size_t height) : AnyImage(image, offset_x, offset_y, width, height) {
	rassert(image.type() == DataTypeTraits<T>::type(), image.type(), DataTypeTraits<T>::type(), 711430549);
}

template <typename T>
TypedImage<T>::TypedImage(const char *const filename) {
    try {
        CImg<T> img(filename);
        CImgWrapper<T> wrapper(img);
        fromCImg(wrapper);
    } catch (CImgIOException& e) {
        init();
        rassert(isNull(), 874294472);
    }
}

template <typename T>
TypedImage<T>::TypedImage(const std::string& filename) : TypedImage(filename.c_str()) {}

template <typename T>
void TypedImage<T>::fromCImg(CImgWrapper<T>& wrapper) {
    CImg<T>& img = *wrapper.img;

    init(img.width(), img.height(), img.spectrum(), DataTypeTraits<T>::type());
    allocateData();

    T* src = img.data();
    for (size_t c = 0; c < cn_; c++) {
        for (size_t y = 0; y < height_; y++) {
            for (size_t x = 0; x < width_; x++) {
                    this->operator()(y, x, c) = *src;
                    ++src;
            }
        }
    }
}

template <typename T>
CImgWrapper<T> TypedImage<T>::toCImg() const {
    CImgWrapper<T> wrapper(width_, height_, cn_);
    CImg<T>& img = *wrapper.img;

    T* dst = img.data();
    for (size_t c = 0; c < cn_; c++) {
        for (size_t y = 0; y < height_; y++) {
            for (size_t x = 0; x < width_; x++) {
                *dst = this->operator()(y, x, c);
                ++dst;
            }
        }
    }

    return wrapper;
}

template <typename T>
TypedImage<T> TypedImage<T>::copy() const {
    TypedImage<T> result(width_, height_, cn_);
    for (size_t y = 0; y < height_; y++) {
        for (size_t x = 0; x < width_; x++) {
            for (size_t c = 0; c < cn_; c++) {
               result(y, x, c) = this->operator()(y, x, c);
            }
        }
    }
    return result;
}

template <typename T>
void TypedImage<T>::fill(T value) {
    for (size_t y = 0; y < height_; y++) {
        for (size_t x = 0; x < width_; x++) {
            for (size_t c = 0; c < cn_; c++) {
                this->operator()(y, x, c) = value;
            }
        }
    }
}

template <typename T>
void TypedImage<T>::fill(T value[]) {
    for (size_t y = 0; y < height_; y++) {
        for (size_t x = 0; x < width_; x++) {
            for (size_t c = 0; c < cn_; c++) {
                this->operator()(y, x, c) = value[c];
            }
        }
    }
}

template <typename T>
void TypedImage<T>::replace(T a, T b) {
    assert (cn_ == 1);
    for (size_t y = 0; y < height_; y++) {
        for (size_t x = 0; x < width_; x++) {
            if (this->operator()(y, x) == a) {
                this->operator()(y, x) = b;
            }
        }
    }
}

template <typename T>
void TypedImage<T>::replace(T a[], T b[]) {
    for (size_t y = 0; y < height_; y++) {
        for (size_t x = 0; x < width_; x++) {
            bool match = true;
            for (size_t c = 0; c < cn_; c++) {
                if (this->operator()(y, x, c) != a[c]) {
                    match = false;
                    break;
                }
            }

            if (match) {
                for (size_t c = 0; c < cn_; c++) {
                    this->operator()(y, x, c) = b[c];
                }
            }
        }
    }
}

class CImgDisplayWrapper {
public:
    CImgDisplay display;
};

ImageWindow::ImageWindow(std::string title) : title(title) {
    cimg_display = std::make_shared<CImgDisplayWrapper>();
    setTitle(title);
}

void ImageWindow::display(const AnyImage &image) {
	     if (image.type() == DataType8u)  { display<unsigned char> (image); }
	else if (image.type() == DataType16u) { display<unsigned short>(image); }
	else if (image.type() == DataType32f) { display<float>         (image); }
	else throwUnsupportedDataType(image.type());
}

template<typename T>
void ImageWindow::display(const TypedImage<T> &image) {
	CImgWrapper<T> wrapper = image.toCImg();
	cimg_display->display.display(*wrapper.img);
	setTitle(title);
}

void ImageWindow::resize(size_t width, size_t height) {
    cimg_display->display.resize(width, height);
}

void ImageWindow::resize() {
    cimg_display->display.resize();
}

void ImageWindow::setTitle(std::string title) {
    cimg_display->display.set_title(title.data());
}

bool ImageWindow::isClosed() {
    return cimg_display->display.is_closed();
}

bool ImageWindow::isResized() {
    return cimg_display->display.is_resized();
}

keycode_t getKeyCode(const char* keycode) {
	return CImgDisplay::keycode(keycode);
}


keycode_t ImageWindow::wait(unsigned int milliseconds) {
    cimg_display->display.wait(milliseconds);
    return cimg_display->display.key();
}

// wait all windows until Escape is pressed or all windows were closed
void waitAllWindows(std::vector<ImageWindow> windows) {
	while (true) {
		bool atLeastOneWindowIsNotClosed = false;
		bool atLeastOneWindowHasEscapePressed = false;
		for (size_t i = 0; i < windows.size(); ++i) {
			ImageWindow &window = windows[i];
			if (!window.isClosed()) {
				atLeastOneWindowIsNotClosed = true;
				if (window.wait(10) == getEscapeKeyCode()) {
					atLeastOneWindowHasEscapePressed = true;
					break;
				}
			}
		}
		if (!atLeastOneWindowIsNotClosed || atLeastOneWindowHasEscapePressed) {
			break;
		}
	}
}

bool isAnyWindowClosed(std::vector<ImageWindow> windows) {
	for (size_t i = 0; i < windows.size(); ++i) {
		if (windows[i].isClosed()) {
			return true;
		}
	}
	return false;
}

bool isEscapePressed(std::vector<ImageWindow> windows, unsigned int milliseconds) {
	for (size_t i = 0; i < windows.size(); ++i) {
		if (windows[i].wait(milliseconds) == getEscapeKeyCode()) {
			return true;
		}
	}
	return false;
}

mouse_click_t ImageWindow::getMouseClick() {
    return cimg_display->display.button();
}

int ImageWindow::getMouseX() {
    return cimg_display->display.mouse_x();
}

int ImageWindow::getMouseY() {
    return cimg_display->display.mouse_y();
}

size_t ImageWindow::width() {
    return cimg_display->display.window_width();
}

size_t ImageWindow::height() {
    return cimg_display->display.window_height();
}

template <typename T>
double TypedImage<T>::sample(double x, double y, size_t c) const {
	// perfect hit in the center of the top-left pixel happens when (x, y) = (0.5, 0.5) 
	x -= 0.5;
	y -= 0.5;

	double dx = x - std::floor(x);
	double dy = y - std::floor(y);

	size_t x0 = std::clamp((ptrdiff_t) std::floor(x), (ptrdiff_t) 0, (ptrdiff_t) width_  - 1);
	size_t y0 = std::clamp((ptrdiff_t) std::floor(y), (ptrdiff_t) 0, (ptrdiff_t) height_ - 1);
	size_t x1 = std::clamp((ptrdiff_t) std::ceil (x), (ptrdiff_t) 0, (ptrdiff_t) width_  - 1);
	size_t y1 = std::clamp((ptrdiff_t) std::ceil (y), (ptrdiff_t) 0, (ptrdiff_t) height_ - 1);

	T a00 = operator()(y0, x0, c); double w00 = (1.0 - dx) * (1.0 - dy);
	T a01 = operator()(y0, x1, c); double w01 = (      dx) * (1.0 - dy);
	T a10 = operator()(y1, x0, c); double w10 = (1.0 - dx) * (      dy);
	T a11 = operator()(y1, x1, c); double w11 = (      dx) * (      dy);

	double interpolated = a00 * w00 + a01 * w01 + a10 * w10 + a11 * w11;
	return interpolated;
}

template <typename T>
T TypedImage<T>::sample_casted(double x, double y, size_t c) const {
	double interpolated = sample(x, y, c);
	if (std::numeric_limits<T>::is_integer) {
		return (T) (interpolated + 0.5);
	} else {
		return (T) interpolated;
	}
}

template void ImageWindow::display<unsigned char>(const TypedImage<unsigned char> &image);
template void ImageWindow::display<unsigned short>(const TypedImage<unsigned short> &image);
template void ImageWindow::display<float>(const TypedImage<float> &image);

template class TypedImage<char>;
template class TypedImage<unsigned char>;
template class TypedImage<short>;
template class TypedImage<unsigned short>;
template class TypedImage<int>;
template class TypedImage<unsigned int>;
template class TypedImage<float>;
template class TypedImage<long long>;
template class TypedImage<unsigned long long>;
template class TypedImage<double>;
