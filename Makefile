.PHONY: build run clean help

# Переменная для имени приложения
APP ?= transpose  # значение по умолчанию
FILE ?= main_matrix_$(APP)

# Основные команды
build:
	mkdir -p build
	cd build && cmake ..
	cd build && make -j8

run: build
	chmod +x ./build/$(FILE)
	./build/$(FILE) 0

# Альтернативные варианты запуска
run-debug: build
	gdb -ex "run" --args ./build/$(FILE) 0

run-valgrind: build
	valgrind ./build/$(FILE) 0

# Очистка
clean:
	rm -rf build

# Справка
help:
	@echo "Доступные команды:"
	@echo "  make build          - Сборка проекта"
	@echo "  make run APP=name   - Сборка и запуск указанного приложения"
	@echo "  make run-debug APP=name - Сборка и запуск под gdb"
	@echo "  make run-valgrind APP=name - Сборка и запуск под valgrind"
	@echo "  make clean          - Очистка build директории"
	@echo "  make help           - Показать эту справку"

# Пример использования:
# make run APP=my_gpu_app
# make run-debug APP=my_gpu_app