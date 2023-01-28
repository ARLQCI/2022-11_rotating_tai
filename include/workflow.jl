using FileIO: FileIO
using JLD2: JLD2


_is_jld2_filename(file) = (file isa String && endswith(file, ".jld2"))

"""
Run some code and write the result to file, or load from the file if it exists.

```julia
run_or_load(file; force=false, verbose=true, kwargs...) do
    data = Dict()  # ...  # something that FileIO can write to `file`
    return data
end
```

runs the code in the block and stores `data` in the given `file`.
If `file` already exists, skip running the code and instead return the data
in `file`.

If `force` is `True`, run the code whether or not `file` exists,
potentially overwriting it.

With `verbose=true`, information about the status of `file` will be shown
as `@info`.

By default, if `file` is a string with a `.jld2` extension, `data` will be
written / read via the `JLD2.save_object` and `JLD2.load_object` functions.
Otherwise, `FileIO.save(file, data; kwargs...)` and `FileIO.load(file)` will be
used by default. These can be overridden by passing a `save` and `load`
function as the second and third positional parameters.

The `data` returned by the code block must be compatible with the format of
`file`. When loading `JLD2.save_object` / `FileIO.load_object`, almost any data
can be written, so this should be particularly safe. More generally, when using
`FileIO.save` and `FileIO.load`, see the [FileIO registry](
https://juliaio.github.io/FileIO.jl/stable/registry/) for details. A common
examples would be a [`DataFrame`](https://dataframes.juliadata.org/stable/)
being written to a `.csv` file.

# See also

* [`@optimize_or_load`](@ref) — for wrapping around [`optimize`](@ref)
* [`DrWatson.@produce_or_load`](https://juliadynamics.github.io/DrWatson.jl/stable/save/#DrWatson.@produce_or_load)
  — a similar but more opinionated function with automatic naming
* [`DrWatson.@tag!`](https://juliadynamics.github.io/DrWatson.jl/stable/save/#DrWatson.@tag!)
  — extend the `data` dict with information about the current file and the
  status of the current git repo
* [`DrWatson.@strdict`](https://juliadynamics.github.io/DrWatson.jl/stable/name/#DrWatson.@strdict)
 — convert multiple values into a `Dict{String,Any}`. This is required for JLD2
 and other formats.
"""
function run_or_load(
    f::Function,
    file;
    save::Function=(_is_jld2_filename(file) ? JLD2.save_object : FileIO.save),
    load::Function=(_is_jld2_filename(file) ? JLD2.load_object : FileIO.load),
    force=false,
    verbose=true,
    kwargs...
)
    # Using `JLD2.save_object` instead of `FileIO.save` is more flexible:
    # `FileIO.save` would require a Dict{String, Any}, whereas `save_object`
    # can save pretty much anything.
    if file isa String
        filename = file
    else
        filename = FileIO.filename(file)
    end
    if force || !isfile(filename)
        if verbose
            if force && isfile(filename)
                @info "Overwriting $filename (force=true)"
            else  # if !isfile(filename)
                @info "File $filename does not exist. Creating it now."
            end
        end
        data = f()
        try
            save(file, data; kwargs...)
        catch
            try
                JLD2.save_object(filename, data)
                msg = "Error saving data. Recover with `using JLD2; load_object($(repr(filename)))`"
                error(msg)
            catch
                error("Error saving data. Unable to recover")
            end
        end
    elseif isfile(filename) && verbose
        @info "Loading data from $filename"
    end
    return load(file)
end
