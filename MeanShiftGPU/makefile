OPENCLC=/System/Library/Frameworks/OpenCL.framework/Libraries/openclc
BUILD_DIR=./build
EXEC_DIR=./bin
EXECUTABLE=MeanShiftGPU
OPENCL_DIR=./OpenCL
.SUFFIXES:
KERNEL_ARCH=i386 x86_64 gpu_32 gpu_64
BITCODES=$(patsubst %, mean_shift_point.cl.%.bc, $(KERNEL_ARCH))

$(EXECUTABLE): $(BUILD_DIR)/mean_shift_point.cl.o $(BUILD_DIR)/main.o $(BITCODES)
	mkdir -p $(EXEC_DIR)
	clang -framework OpenCL -o $@ $(BUILD_DIR)/mean_shift_point.cl.o $(BUILD_DIR)/main.o -L/usr/local/include -L/usr/local/lib -lgsl -lgslcblas

$(BUILD_DIR)/mean_shift_point.cl.o: mean_shift_point.cl.c
	mkdir -p $(BUILD_DIR)
	clang -c -Os -Wall -arch x86_64 -o $@ -c mean_shift_point.cl.c

$(BUILD_DIR)/main.o: main.c mean_shift_point.cl.h
	mkdir -p $(BUILD_DIR)
	clang -c -Os -Wall -arch x86_64 -o $@ -c $<

mean_shift_point.cl.c mean_shift_point.cl.h: mean_shift_point.cl
	$(OPENCLC) -x cl -cl-std=CL1.1 -cl-auto-vectorize-enable -emit-gcl $<

mean_shift_point.cl.%.bc: mean_shift_point.cl
	$(OPENCLC) -x cl -cl-std=CL1.1 -Os -arch $* -emit-llvm -o $@ -c $<

.PHONY: clean
clean:
	mkdir -p $(EXEC_DIR)
	mv $(EXECUTABLE) $(EXEC_DIR)/
	mv *.bc $(EXEC_DIR)/
	rm -rf $(BUILD_DIR) mean_shift_point.cl.h mean_shift_point.cl.c $(EXECUTABLE) *.bc
