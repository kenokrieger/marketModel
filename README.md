# Ising Market Model

<img src="https://img.shields.io/github/issues/kenokrieger/marketModel"> <img src="https://img.shields.io/github/commit-activity/m/kenokrieger/marketModel">
<img src="http://qmpy.org/badges/license.svg">

Modified source code from the GPU Ising Model built by
<a href="https://github.com/romerojosh">Joshua Romero</a>.

## Outline

This particular model attempts to predict the behaviour of traders in a market
governed by two simple guidelines:

- Do as neighbours do

- Do what the minority does

mathematically speaking is each trader represented by a spin on a two dimensional
grid. The local field of each spin *S*<sub>i</sub> is given by the equation below

<img src="https://render.githubusercontent.com/render/math?math=h_i(t) = \sum_{j = 1}^N J_{ij} S_j - \alpha S_i \left| \frac{1}{N} \sum_{j = 1}^N S_j \right|">

where *J*<sub>ij</sub> = *j* for the nearest neighbours and 0 otherwise. The spins
are updated according to a Heatbath dynamic which reads as follows

<img src="https://render.githubusercontent.com/render/math?math=S_i(t %2B 1) = %2B 1 \quad \mathrm{with} \quad p = \frac{1}{1 %2B \exp(-2\beta h_i(t))}">
<img src="https://render.githubusercontent.com/render/math?math=S_i(t %2B 1) = -1 \quad \mathrm{with} \quad 1 - p">

The model is thus controlled by the three parameters

- &alpha;, which represents the tendency of the traders to be in the minority

- *j*, which affects how likely it is for a trader to pick up the strategy of its neighbour

- &beta;, which controls the randomness

In each iteration all spins are updated in parallel using the metropolis
algorithm. </br>
(For more details see <a href="https://arxiv.org/pdf/cond-mat/0105224.pdf">
S.Bornholdt, "Expectation bubbles in a spin model of markets: Intermittency from
frustration across scales, 2001"</a>)

## Compiling

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
Then compile the code with the nvcc-compiler. To compile the simulation use

```terminal
nvcc -l curand -I common\inc -I Libraries\include --library-path Libraries\lib kernel.cu ProgressBar.cpp -o build\model.exe
```

For the thread speed test use

```terminal
nvcc -l curand thread_test.cu -o build\threads.exe
```

Additional libraries used are specified through the -l option. Additional include
paths are specified with the -I option.

OpenGl will fail to create a window when used in a remote desktop session. To
use the program in a remote session, write a batch script that first closes the
session

```terminal
tscon rdp-tcp#YOUR_SESSION_ID /dest:console
```

and the runs the program by simply calling it from the command line. Then simply reconnect to the remote desktop.
You can get your session id by running

```terminal
query session
```

in the terminal. </br>
Note that you need to execute the batch file as an administrator.

## Running the executable

**Note**: Command line arguments are currently not supported which means if you want
to change some of the parameters you need to change their values in the code itself.</br>

To successfully run the program, it needs to find the 'freeglut.dll' file which
you can download from http://freeglut.sourceforge.net. <br/>

Upon running the executable for the simulation you may adjust the parameters, get a live
view or save the current configuration to a file, by pressing specific buttons (see table below).

| key      | function                                              |
|----------|-------------------------------------------------------|
| esc      | Save the current state and close the simulation       |
| spacebar | Toggle liew view on and off                           |
| s        | Save the current state to a file                      |
| c        | Change the parameters                                 |
| i        | Prints out information on the state of the simulation |
| p        | Pause the simulation                                  |

To visualise some of the configurations saved to a file simply invoke the
python script 'visualise.py' with the path to the save file(s) as command line
arguments.

## Profiling

Need to add to path

```terminal
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\extras\CUPTI\lib64
```

```terminal
nvprof --log-file threads.profile threads.exe
```
