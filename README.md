# Simulations: Rotating Sensing Using Tractor Atom Interferometry

## Prerequisites

* Julia 1.8.5 (install with [juliaup](https://github.com/JuliaLang/juliaup))
* [IJulia](https://github.com/JuliaLang/IJulia.jl) installed into the main Julia 1.8.5 environment. Make sure the kernel is properly installed by running `using IJulia; installkernel("Julia", "--project=@.")` in the Julia REPL.
* [Jupyter Lab](https://jupyter.org) with [jupytext extension](https://jupytext.readthedocs.io/en/latest/). Recommended to install via [miniforge](https://github.com/conda-forge/miniforge):

  ```
  conda install jupyterlab notebook jupyterlab_execute_time nbdime jupytext ipywidgets ipyparallel python-localvenv-kernel markdown-kernel
  ```

## Environment instantiation

* Run `julia --project=.` in a checkout of this project and then `] instantiate` in the resulting REPL.
* Consider running `make -j4 ipynb` (where `4` is the number of available cores) to create missing `.ipynb` files from the available `.jl` notebook files. Or `make <name of notebook.ipynb>` to create a particular notebook. Generating all notebooks may take several hours, so this is best run overnight.

## Notebooks

* In a terminal, from a checkout of this project, run `JULIA_NUM_THREADS=auto jupyter lab`
* The `jupytext` extension ensures that the `.jl` files in the root folder can be opened as notebooks. Saving a notebook will automatically create a corresponding `.ipynb` file. In principle, the `.jl` and `.ipynb` files can be used interchangeable, although some features of Jupyter work better when opening the `.ipynb` files.

See [#INDEX.md](%23INDEX.md) for an overview of the notebooks. Assuming `make ipynb` has been run, the links in that file can be used to open any of the notebooks.
