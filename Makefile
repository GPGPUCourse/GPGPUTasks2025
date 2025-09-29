# Makefile для сборки и запуска GPU проекта

.PHONY: build run clean help

# Основные команды
build:
	mkdir -p build
	cd build && cmake ..
	cd build && make -j8

run: build
	./build/main_aplusb_matrix 0

# Альтернативные варианты запуска
run-debug: build
	gdb -ex "run" --args ./build/main_aplusb_matrix 0

run-valgrind: build
	valgrind ./build/main_aplusb_matrix 0

# Очистка
clean:
	rm -rf build

# Справка
help:
	@echo "Доступные команды:"
	@echo "  make build     - Очистка, создание build директории и сборка проекта"
	@echo "  make run       - Сборка и запуск с аргументом 0"
	@echo "  make run-debug - Сборка и запуск под gdb"
	@echo "  make run-valgrind - Сборка и запуск под valgrind"
	@echo "  make clean     - Очистка build директории"
	@echo "  make help      - Показать эту справку"

# Убедитесь что используется tab для отступов, а не пробелы!