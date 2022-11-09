import QuantumPropagators.Controls:
    getcontrols, evalcontrols, substitute_controls
import QuantumControlBase: getcontrolderiv


mutable struct IntegratedAmplitudeEvalCache
    n::Int64
    val::Float64
end

struct IntegratedAmplitude
    control :: Vector{Float64}
    eval_cache :: IntegratedAmplitudeEvalCache
    function IntegratedAmplitude(control)
        new(control, IntegratedAmplitudeEvalCache(0, 0.0))
    end
end


function evalcontrols(ampl::IntegratedAmplitude, vals_dict, tlist, n)
    control = ampl.control
    if length(control) ≠ (length(tlist) - 1)
        error("control must be defined on the intervals of tlist")
    end
    val = 0.0
    #if ampl.eval_cache.n == n-1
        #val = ampl.eval_cache.val
    #else
        for i ∈ 1:n-1
            dt = tlist[i+1] - tlist[i]
            val += control[i] * dt
        end
    #end
    dt = tlist[n+1] - tlist[n]
    val += get(vals_dict, control, control[n]) * dt
    ampl.eval_cache.n = n
    ampl.eval_cache.val = val
    if isnan(val) # DEBUG
        @show dt
        @show vals_dict[control]
        @show any(isnan.(control))
        error("ϕ is NaN")
    end
    return val
end


function getcontrols(ampl::IntegratedAmplitude)
    return (ampl.control, )
end


function substitute_controls(ampl::IntegratedAmplitude, controls_map)
    return IntegratedAmplitude(get(controls_map, control, control))
end


function getcontrolderiv(ampl::IntegratedAmplitude, control)
    if control ≡ ampl.control
        return 1.0
    else
        return 0.0
    end
end
