#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

template<typename T>
std::string to_string(T value)
{
	std::ostringstream ss;
	ss << value;
	return ss.str();
}

void reportError(cl_int err, const std::string &filename, int line)
{
	if(CL_SUCCESS == err)
		return;

	// Таблица с кодами ошибок:
	// libs/clew/CL/cl.h:103
	// P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
	std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
	throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

int main()
{
	// Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку libs/clew)
	if(!ocl_init())
		throw std::runtime_error("Can't init OpenCL driver!");

	// Откройте
	// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/
	// Нажмите слева: "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformIDs"
	// Прочитайте документацию clGetPlatformIDs и убедитесь, что этот способ узнать, сколько есть платформ, соответствует документации:
	cl_uint platformsCount = 0;
	OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
	std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

	// Тот же метод используется для того, чтобы получить идентификаторы всех платформ - сверьтесь с документацией, что это сделано верно:
	std::vector<cl_platform_id> platforms(platformsCount);
	OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

	for(int platformIndex = 0; platformIndex < platformsCount; ++platformIndex)
	{
		std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
		cl_platform_id platform = platforms[platformIndex];

		// Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
		// Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL
		size_t platformNameSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
		// TODO 1.1
		// Попробуйте вместо CL_PLATFORM_NAME передать какое-нибудь случайное число - например 239
		// Т.к. это некорректный идентификатор параметра платформы - то метод вернет код ошибки
		// Макрос OCL_SAFE_CALL заметит это, и кинет ошибку с кодом
		// Откройте таблицу с кодами ошибок:
		// libs/clew/CL/cl.h:103
		// P.S. Быстрый переход] к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
		// Найдите там нужный код ошибки и ее название
		// Затем откройте документацию по clGetPlatformInfo и в секции Errors найдите ошибку, с которой столкнулись
		// в документации подробно объясняется, какой ситуации соответствует данная ошибка, и это позволит, проверив код, понять, чем же вызвана данная ошибка (некорректным аргументом param_name)
		// Обратите внимание, что в этом же libs/clew/CL/cl.h файле указаны всевоможные defines, такие как CL_DEVICE_TYPE_GPU и т.п.

//		OCL_SAFE_CALL(clGetPlatformInfo(platform, 239, 0, nullptr, &platformNameSize));
		// ошибка вылазит -30 потому что парам нейм должен вернуть один из конкретных пуллов значений который мы запросим, по типу название платформы, а он принял просто число какое то
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize)); // вернул норм парам нейм чтобы дальше проверять

		// TODO 1.2
		// Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:

		std::vector<unsigned char> platformName(platformNameSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
		std::cout << "    Platform name: " << platformName.data() << std::endl;

		// вывел apple

		// TODO 1.3
		// Запросите и напечатайте так же в консоль вендора данной платформы

		size_t platformVendorSize = 0;
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platformVendorSize));
		std::vector<unsigned char> platformVendor(platformVendorSize, 0);
		OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendorSize, platformVendor.data(), nullptr));
		std::cout << "Platform vendor: " << platformVendor.data() << std::endl;

		// получается что в 1 задании мы получали инфу через след пайплайн: узнаем размер ответа (например длина названии вендора) через clGetPlatformInfo просто не копируя ничего,
		// а указывая куда записать полученный ответ от param_name. потом уже создавали вектор чаров нужной длины и уже туда копировали ответ. все еще обворачивали макросом чтобы поймать ошибку

		// TODO 2.1
		// Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")

		cl_uint devicesCount = 0;
		cl_int numOfDevices = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount);
		if (numOfDevices == CL_DEVICE_NOT_FOUND)
			continue;
		else
			OCL_SAFE_CALL(numOfDevices);

		// создается переменная devicesCount которая будет счетчиков кол-ва девайсов на платформе.
		//  мы идем и обращаемся к clGetDeviceIds и присваиваем счетчику каждый раз +1 если девайс либо цпу либо гпу.
		// затем мы хендлил если вдруг нет девайса это ок но если там какая то инородная фигня кидаем макрос.


		std::vector<cl_device_id> devices(devicesCount);
		OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));
		for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
		{
			// TODO 2.2
			// Запросите и напечатайте в консоль:
			// - Название устройства
			// - Тип устройства (видеокарта/процессор/что-то странное)
			// - Размер памяти устройства в мегабайтах
			// - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными

			// тут так же везде такая же логика - создаем переменную нужного типа, вызываем clGetDeviceInfo с нужным параметром и сохраняем, хендлим ошибку макросом

			// вывожу номер девайса а затем название
			cl_device_id device = devices[deviceIndex];
			std::cout << "Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
			size_t deviceNameSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
			std::vector<unsigned char> deviceName(deviceNameSize, 0);
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
			std::cout << "Device name: " << deviceName.data() << std::endl;

			// выводим тип устройства просто условиями нахожу что у меня
			cl_device_type deviceType = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr));
			std::string deviceTypeStr;
			if (deviceType & CL_DEVICE_TYPE_CPU)
				deviceTypeStr += "CPU ";
			if (deviceType & CL_DEVICE_TYPE_GPU)
				deviceTypeStr += "GPU ";
			if (deviceType & CL_DEVICE_TYPE_ACCELERATOR)
				deviceTypeStr += "ACCELERATOR ";
			if (deviceType & CL_DEVICE_TYPE_DEFAULT)
				deviceTypeStr += "DEFAULT ";
			if (deviceType & CL_DEVICE_TYPE_CUSTOM)
				deviceTypeStr += "CUSTOM ";
			std::cout << "Device type: " << deviceTypeStr << std::endl;

			// вывожу размер памяти в мегабайтах (поэтому и делю на 1024^^2)
			cl_ulong globalMemSize = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMemSize), &globalMemSize, nullptr));
			std::cout << "Global memory size: " << (globalMemSize / (1024 * 1024)) << " MB" << std::endl;

			// считаю и вывожу кол-во ядер (как доп свойство полезно например чтобы понять кол-во параллельных вычислений которые может выполнять устройство)
			cl_uint maxComputeUnits = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, nullptr));
			std::cout << "Max compute units: " << maxComputeUnits << std::endl;

			// вывожу макс частоту процессора в мегагерцах (ну тоже полезно знать, хотя под вопросом но мне это пришло в голову)) )
			cl_uint maxClockFrequency = 0;
			OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(maxClockFrequency), &maxClockFrequency, nullptr));
			std::cout << "Max clock frequency: " << maxClockFrequency << " MHz" << std::endl;
		}
	}

	return 0;
}
