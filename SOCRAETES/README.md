## SOCRAETES: SOlar Cells Recorded And EmulaTed EaSily

**Sayedsepehr Mosavat, Matteo Zella, Pedro José Marrón**  
**Networked Embedded Systems Group (NES) - Universität Duisburg-Essen**

@copyright https://github.com/SepehrMosavat/SOCRAETES

SOCRAETES is a tool for supporting the design and evaluation of energy harvesting
systems. The core idea behind the development of SOCRAETES has been to
enable easy replication of the required hardware components, therefore
decreasing the entry-barrier into this field of research, even for researchers
and hobbyists that have little prior experience with electronics. This is
achieved by simplifying the hardware design as much as possible, and also by
employing only through-hole components in the required hardware.

The hardware design of SOCRAETES provides a high level of flexibility in terms
of the range of measurements and emulations. Since there is a tradeoff between
this range and the overall accuracy of the system, the users have the
possibility of modifying the underlying hardware by using different passive
components. By doing so, each user can target the best overall range of
measurements and emulation, while also achieving a high level of accuracy.

Please refer to our demo abstract, presented at EWSN 2021, for more information
regarding the design and implementation of SOCRAETES.

To use SOCRAETES, the following prerequisites exist:
- SOCRAETES hardware, either assembled on a prototyping board, or optionally,
using the PCB found in the hardware directory. Please note that the full
functionality of SOCRAETES can be realized using prototyping boards, but the
highest reliability and accuracy will be achieved only if the PCB is used.
Please refer to the hardware directory for instructions regarding preparing the
hardware
- SOCRAETES firmware, which will be running on a Teensy 3.6. Please refer to
the firmware directory for instructions on flashing the Teensy board.
- SOCRAETES software, which will be running on a host PC and used for
acquiring data and emulating traces. The current implementation of SOCRAETES
is using Python code, but we are working on extensions of this functionality
for further data visualization and analysis.

#### Contact us
In case you experience any issues while using SOCRAETES, or have any questions
or ideas for further improvements, please do not hesitate to contact us via
e-mail! You can find our contact information on our [web page](https://nes.uni-due.de).

## SOCRAETES: SOlar Cells Recorded And EmulaTed EaSily

This document provides the instructions for setting up and using the
software of SOCRAETES. The software is meant for being run on a host system,
to which the SOCRAETES hardware is connected. The software, developed in Python,
will communicate with the hardware and provide a user interface for functionalities
such as data visualization and retention.

The current software provides the user a command line interface to either record or emulate
energy harvesting traces of a solar cell. Either functionality can be used
as follows.

#### Requirements
The dependencies of the Python code can be found in *requirements.txt*. The
dependencies can be installed using the following command:
```
pip install -r requirements.txt
```
#### Recording
The following examples demonstrate how to record traces in different modes.
To record a trace and visualize it using a 2D plot:
```
record.py --port <Serial Port>
```
The only necessary parameter is the serial port, on which the hardware is
communicating with the host system. Not providing further parameters will
record a trace in the default operation mode, which is 2D plotting of the trace.

The above example could also be run explicitly using the following command:
```
record.py --port <Serial Port> --mode plot-curve
```
Alternatively, the trace can be visualized with a 3D surface by using the following
command:
```
record.py --port <Serial Port> --mode plot-surface
```
Furthermore, a trace can be captured for 30 seconds and saved on the local
machine in an HDF5 file using the following command:
```
record.py --port <Serial Port> --mode commit-to-file --duration 30
```
The command above will automatically generate a unique file name for the output
file. However, the file name can be chosen explicitly by the user using the
following command:
```
record.py --port <Serial Port> --mode commit-to-file --file <File Name> --duration 30
```
Last but not least, the energy harvesting environment can be described by
parameters that will be appended to the output file as metadata. Such data can
be used for data analysis in later stages. For example, the following command
will capture 30 seconds of data in an outdoor situation on a sunny day, in
Berlin, Germany. The ambient light intensity of the environment is estimated
to be 150 Lux:
```
record.py --port <Serial Port> --mode commit-to-file --duration 30 --environment indoor
--lux 150 --weather sunny --country Germany --city Berlin
```
#### Emulation
The following examples demonstrate how to emulate energy harvesting traces using
SOCRAETES. The emulation can be performed either by using a file containing
previously-recorded traces or by using a user-defined array with arbitrary
data.

The following command will emulate a trace from a file:
```
emulate.py --port <Serial Port> --source file --file <File Name>
```
On the other hand, the following command can be used for emulating a user-defined
set of operation parameters:
```
emulate.py --port <Serial Port> --source array --array [[0.5, 0], [1, 1000], [2, 2000]]
```
The user-defined array has the following format: ``[[<DELAY BETWEEN CURVES (s)>,0],[OPEN CIRCUIT VOLTAGE (V)>,<SHORT CIRCUIT CURRENT (uA)>],...]``
In the previous example two curves will the emulated with a delay of 0.5 seconds
between each. The first curve will have an open-circuit voltage of 1V and a
short-circuit current of 1000uA. The second curve will have an open-circuit
voltage of 2V and a short-circuit current of 2000uA.
