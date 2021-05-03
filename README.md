Modified source code from the Ising Model built by
<a href="https://github.com/romerojosh">Joshua Romero</a>.

# Compiling

To compile the code on Windows 10 you need to have Microsoft Visual Studio
2019 installed. With that comes the file vcvarsall.bat which is needed to
properly set up the development environment. The compiler needed for the cuda
code is the nvcc-compiler which is included in the CUDA Toolkit.
Assuming you want to create an executable file called "model.exe" in the build
directory, open a command prompt in the directory where the .cu source files are
located and execute the vcvarsall.bat file, e.g.:

```terminal
"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall" x86_amd64
```

This should configure the terminal to find all paths needed for compilation.
Then compile the code with the nvcc-compiler.

```terminal
nvcc -l curand -I common\inc kernel.cu ProgressBar.cpp -o build\model.exe
```

The -l option tells the compiler that the curand library is used. The -I option
specifies the path to external dependencies. The resulting file will be placed
in the build directory.
