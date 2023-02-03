using QuantumPropagators.Amplitudes: ControlAmplitude
using QuantumControlBase.PulseParametrizations: ShapedParametrizationPulseDerivative
using QuantumPropagators.Controls: discretize_on_midpoints

import QuantumPropagators.Controls: evaluate
import QuantumControlBase: get_control_deriv


abstract type PotScalingAmplitude <: ControlAmplitude end


function PotScalingAmplitude(control; shape)
    if (control isa Vector{Float64}) && (shape isa Vector{Float64})
        return PotScalingPulseAmplitude(control, shape, guide)
    else
        try
            ϵ_t = control(0.0)
        catch
            error(
                "A PotScalingAmplitude control must either be a vector of values or a callable"
            )
        end
        try
            S_t = shape(0.0)
        catch
            error("A PotScalingAmplitude shape must either be a vector of values or a callable")
        end
        return PotScalingContinuousAmplitude(control, shape)
    end
end


function PotScalingAmplitude(control, tlist; shape)
    control = discretize_on_midpoints(control, tlist)
    shape = discretize_on_midpoints(shape, tlist)
    return PotScalingPulseAmplitude(control, shape)
end


function Base.show(io::IO, ampl::PotScalingAmplitude)
    print(
        io,
        "PotScalingAmplitude(::$(typeof(ampl.control)); shape::$(typeof(ampl.shape)))"
    )
end


struct PotScalingPulseAmplitude <: PotScalingAmplitude
    control::Vector{Float64}
    shape::Vector{Float64}
end


function Base.Array(ampl::PotScalingPulseAmplitude)
    1.0 .+ ampl.shape .* ampl.control.^2
end


struct PotScalingContinuousAmplitude <: PotScalingAmplitude
    control
    shape
end


function (ampl::PotScalingContinuousAmplitude)(t::Float64)
    return 1.0 + ampl.shape(t) * ampl.control(t)^2
end


function evaluate(ampl::PotScalingPulseAmplitude, args...; kwargs...)
    S = evaluate(ampl.shape, args...; kwargs...)
    ϵ = evaluate(ampl.control, args...; kwargs...)
    return 1.0 + S * ϵ^2
end


function evaluate(ampl::PotScalingContinuousAmplitude, args...; kwargs...)
    error("Not implemented")
end


function get_control_deriv(ampl::PotScalingAmplitude, control)
    if control ≡ ampl.control
        return ShapedParametrizationPulseDerivative(ampl.control, ϵ->2ϵ, ampl.shape)
    else
        0.0
    end
end
