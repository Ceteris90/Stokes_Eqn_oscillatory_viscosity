# Stokes Equation with Oscillatory Viscosity

This repository contains a C++ numerical simulation of the **Stokes Equation** featuring a spatially oscillatory viscosity profile. This project is an educational extension of the **step-7** example from the [deal.II library](https://www.dealii.org/).

## 📖 Overview

The simulation solves the incompressible Stokes equations for velocity $\mathbf{u}$ and pressure $p$:

$$- \nabla \cdot ( \mu(\mathbf{x}) \nabla \mathbf{u} ) + \nabla p = \mathbf{f}$$
$$\nabla \cdot \mathbf{u} = 0$$

Where $\mu(\mathbf{x})$ represents the **oscillatory viscosity coefficient**. This setup is particularly useful for studying fluid behavior in heterogeneous media or varying thermal environments.

---

## 🛠 Prerequisites

Before building, ensure you have the following installed:

* **OS:** A Linux distribution (Ubuntu, Debian, Fedora, etc.)
* **Compiler:** A C/C++ compiler (GCC or Clang)
* **Build System:** [CMake](https://cmake.org/) v2.8.12 or higher
* **Library:** [deal.II](https://www.dealii.org/) v9.3.1 or higher
* **Visualization:** [ParaView](https://www.paraview.org/) to view `.vtk` output files

---

## 🚀 Building the Program

We recommend an **out-of-source build** to keep the repository clean.

### 1. Clone the repository
```bash
git clone https://github.com/Ceteris90/Stokes_Eqn_oscillatory_viscosity.git
```

### 2. Configure with CMake
Create a build directory parallel to the source code and generate the build files (including Eclipse CDT project files if desired):

```bash
mkdir build-Stokes_Eqn_oscillatory_viscosity
cd build-Stokes_Eqn_oscillatory_viscosity

# Replace /path/to/dealii with your actual deal.II installation path
# Replace N with the number of CPU cores
cmake -DDEAL_II_DIR=/path/to/dealii -DCMAKE_ECLIPSE_MAKE_ARGUMENTS=-jN -G"Eclipse CDT4 - Unix Makefiles" ../Stokes_Eqn_oscillatory_viscosity
```

### 3. Compile
You can compile in either **Debug** mode (for error checking) or **Release** mode (for speed).

* **For Debug:**
    ```bash
    make debug
    make -jN
    ```
* **For Release:**
    ```bash
    make release
    make -jN
    ```

---

## 🏃 Running and Visualization

### Execution
Run the solver using the default executable:
```bash
./Stokes_Equation
```

### Viewing Results
The simulation produces `.vtk` files. To visualize the fluid flow:
1.  Open **ParaView**.
2.  Load the generated `.vtk` files.
3.  Apply a **Glyph** filter to see velocity vectors or a **Surface** map for pressure distribution.

---

> **Note:** This code is primarily for educational purposes, demonstrating how to modify standard finite element examples to handle variable coefficients in fluid dynamics.
