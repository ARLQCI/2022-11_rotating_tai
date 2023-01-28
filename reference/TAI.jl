using LinearAlgebra
using FFTW
using Zygote

FFTW.set_provider!("mkl")

const μm = 1
const μs = 1
const cm = 1e4μm
const met = 1e6μm
const sec = 1e6μs
const ms = 1e3μs
const MHz = 2π
const Dalton = 1.5746097504353806e+01
const Rb_mass = 86.91Dalton;

using Plots
using Plots.PlotMeasures: mm

Plots.default(
    linewidth=3,
    size=(550, 300),
    framestyle=:box,
    background_color=RGBA(1, 1, 1, 0),
    foreground_color=:black
)

function plot_dyn(wrk, zlist)

    heatmap(
        wrk.zgrid / μm,
        wrk.tlist / ms,
        abs2.(wrk.Ψ_list'),
        # xlim=(-10, 60),
        xlabel="z (μm)",
        ylabel="time (ms)",
        colorbar_title="\nProbability density",
        color=cgrad(["#fbf8efff", :blue, :purple], [0, 0.2, 0.5]),
        right_margin=10mm
    )

    plot!(zlist / μm, tlist / ms, label="", color=:black, linestyle=:dash)
end

mutable struct Wrk
    zmin
    zmax
    nz
    zgrid
    pgrid
    K
    to_p!
    to_x!
    to_p
    to_x
    tlist
    h
    eval_V!
    eval_dV!
    eval_Ux!
    eval_dUx!
    Ψ_list
    grad
    Vx
    dVx
    Uk2
    Uk2b
    χx
    Ψaux
    function Wrk(zmin, zmax, tlist, V, dV; nz=1024, Rb_mass=Rb_mass)

        h = tlist[2] - tlist[1]

        zgrid = collect(range(zmin, zmax, nz))
        dz = zgrid[2] - zgrid[1]

        pgrid = 2π * fftfreq(nz, 1 / dz)
        K = pgrid .^ 2 / 2 / Rb_mass

        Ψrand = rand(ComplexF64, nz)
        normalize!(Ψrand)

        to_p! = plan_fft!(Ψrand)
        to_x! = plan_ifft!(Ψrand)
        to_p = plan_fft(Ψrand)
        to_x = plan_ifft(Ψrand)

        function eval_V!(Vx, z0)
            Vx .= V.(zgrid; z0=z0)
        end

        function eval_dV!(Vx, z0)
            Vx .= dV.(zgrid; z0=z0)
        end

        function eval_Ux!(Ψ, z0; h=h)
            for i = 1:nz
                @inbounds Ψ[i] *= exp(-1im * h * V(zgrid[i]; z0=z0))
            end
        end

        function eval_dUx!(Ψ, z0)
            for i = 1:nz
                @inbounds Ψ[i] *=
                    -1im * h * exp(-1im * h * V(zgrid[i]; z0=z0)) * dV(zgrid[i]; z0=z0)
            end
        end

        Ψ_list = zeros(ComplexF64, nz, length(tlist))
        grad = zeros(Float64, length(tlist))
        Vx = zeros(Float64, nz)
        dVx = zeros(Float64, nz)
        Uk2 = exp.(-0.5im * h * K)
        Uk2b = exp.(+0.5im * h * K)
        χx = zeros(ComplexF64, nz)
        Ψaux = zeros(ComplexF64, nz)

        new(
            zmin,
            zmax,
            nz,
            zgrid,
            pgrid,
            K,
            to_p!,
            to_x!,
            to_p,
            to_x,
            tlist,
            h,
            eval_V!,
            eval_dV!,
            eval_Ux!,
            eval_dUx!,
            Ψ_list,
            grad,
            Vx,
            dVx,
            Uk2,
            Uk2b,
            χx,
            Ψaux
        )
    end
end;

function get_ground_state(wrk, z0=0; steps=10000, d=1.0, h0=1)
    wrk.eval_V!(wrk.Vx, z0)
    h = -1im
    Uk2 = exp.(-0.5im * h * wrk.K)
    Uk = exp.(-1im * h * wrk.K)
    Ux = exp.(-1im * h * wrk.Vx)

    Ψx = convert(Array{ComplexF64}, exp.(-(wrk.zgrid .- z0) .^ 2 / d^2))
    normalize!(Ψx)

    Ψk = fft(Ψx)

    for i = 1:steps
        Ψk = Uk2 .* Ψk
        Ψx = ifft(Ψk)
        Ψx = Ux .* Ψx
        Ψk = fft(Ψx)
        Ψk = Uk2 .* Ψk

        normalize!(Ψk)
    end

    Ψx = ifft(Ψk)
    normalize!(Ψx)

    return Ψx
end

function step!(wrk, Ψ, z0; h=wrk.h)
    wrk.to_p! * Ψ
    Ψ .*= ifelse.(h > 0, wrk.Uk2, wrk.Uk2b)
    wrk.to_x! * Ψ
    wrk.eval_Ux!(Ψ, z0; h=h)
    wrk.to_p! * Ψ
    Ψ .*= ifelse.(h > 0, wrk.Uk2, wrk.Uk2b)
    wrk.to_x! * Ψ
end;

function dstep!(wrk, Ψ, z0)
    wrk.to_p! * Ψ
    Ψ .*= wrk.Uk2
    wrk.to_x! * Ψ
    wrk.eval_dUx!(Ψ, z0)
    wrk.to_p! * Ψ
    Ψ .*= wrk.Uk2
    wrk.to_x! * Ψ
end

function propagate(wrk, Ψx0, zlist; save=true)
    Ψx = copy(Ψx0)
    if save
        for (i, z0) in enumerate(zlist)
            step!(wrk, Ψx, z0)
            wrk.Ψ_list[:, i] .= Ψx
        end
    else
        for z0 in zlist
            step!(wrk, Ψx, z0)
        end
    end
    Ψx
end;

function g(wrk, Ψx0, Ψxt, zlist)
    g_val = 0
    propagate(wrk, Ψx0, zlist)
    for i = 1:length(zlist)
        g_val -= abs2(Ψxt' * @view wrk.Ψ_list[:, i])
    end

    return g_val / length(wrk.tlist)
end;

function f(wrk, Ψx0, Ψxt, zlist)
    propagate(wrk, Ψx0, zlist)
    -abs2(Ψxt' * @view wrk.Ψ_list[:, end])
end;

function df(wrk, Ψx0, Ψxt, zlist; acc=false, cal_val=false)

    if cal_val
        f_val = g(wrk, Ψx0, Ψxt, zlist)
    else
        propagate(wrk, Ψx0, zlist)
    end

    z0 = zlist[end]
    wrk.χx .= 2 * (Ψxt' * (@view wrk.Ψ_list[:, end])) * Ψxt

    for i = length(zlist)-1:-1:2
        step!(wrk, wrk.χx, z0; h=-wrk.h)
        z0 = zlist[i]
        wrk.Ψaux .= @view wrk.Ψ_list[:, i-1]
        dstep!(wrk, wrk.Ψaux, z0)
        if !acc
            wrk.grad[i] = 0
        end
        wrk.grad[i] += real(wrk.χx' * wrk.Ψaux) / length(tlist)
    end

    if cal_val
        return f_val
    end

end;

function df_n(wrk, Ψx0, Ψxt, zlist, i; ϵ=1e-7)
    zlist_p = copy(zlist)
    zlist_m = copy(zlist)
    zlist_p[i] += ϵ
    zlist_m[i] -= ϵ

    (f(wrk, Ψx0, Ψxt, zlist_p) - f(wrk, Ψx0, Ψxt, zlist_m)) / 2ϵ
end;

function dg_n(wrk, Ψx0, Ψxt, zlist, i; ϵ=1e-7)
    zlist_p = copy(zlist)
    zlist_m = copy(zlist)
    zlist_p[i] += ϵ
    zlist_m[i] -= ϵ

    (g(wrk, Ψx0, Ψxt, zlist_p) - g(wrk, Ψx0, Ψxt, zlist_m)) / 2ϵ
end;

function dg(wrk, Ψx0, Ψxt, zlist; acc=false, cal_val=false)
    if cal_val
        g_val = g(wrk, Ψx0, Ψxt, zlist)
    else
        propagate(wrk, Ψx0, zlist)
    end

    z0 = zlist[end]
    wrk.χx .= 2 * (Ψxt' * (@view wrk.Ψ_list[:, end])) * Ψxt

    for i = length(zlist)-1:-1:2
        step!(wrk, wrk.χx, z0; h=-wrk.h)
        z0 = zlist[i]
        wrk.χx .+= 2 * (Ψxt' * (@view wrk.Ψ_list[:, i])) * Ψxt
        wrk.Ψaux .= @view wrk.Ψ_list[:, i-1]
        dstep!(wrk, wrk.Ψaux, z0)
        if !acc
            wrk.grad[i] = 0
        end
        wrk.grad[i] += real(wrk.χx' * wrk.Ψaux) / length(tlist)
    end

    if cal_val
        return g_val
    end
end;

Zygote.@adjoint function g(wrk, Ψx0, Ψxt, zlist)

    g_val = 0
    propagate(wrk, Ψx0, zlist)
    for i = 1:length(zlist)
        g_val -= abs2(Ψxt' * @view wrk.Ψ_list[:, i])
    end

    function g_pullback(Ω̄)
        z0 = zlist[end]
        wrk.χx .= 2 * (Ψxt' * (@view wrk.Ψ_list[:, end])) * Ψxt

        for i = length(zlist)-1:-1:2
            step!(wrk, wrk.χx, z0; h=-wrk.h)
            z0 = zlist[i]
            wrk.χx .+= 2 * (Ψxt' * (@view wrk.Ψ_list[:, i])) * Ψxt
            wrk.Ψaux .= @view wrk.Ψ_list[:, i-1]
            dstep!(wrk, wrk.Ψaux, z0)
            wrk.grad[i] = -real(wrk.χx' * wrk.Ψaux) / length(tlist)
        end

        return nothing, nothing, nothing, Ω̄' * wrk.grad
    end
    return g_val / length(tlist), g_pullback
end

function u(wrk, zlist; α=1)
    u_val = 0
    for i = 1:length(zlist)-1
        u_val += (zlist[i+1] - zlist[i])^2
    end
    α * u_val / length(zlist) #/ wrk.h^2
end

function du_n(wrk, zlist, i; ϵ=1e-6)
    zlist_p = 1.0 .* zlist
    zlist_m = 1.0 .* zlist
    zlist_p[i] += ϵ
    zlist_m[i] -= ϵ

    @show u(wrk, zlist_p)
    @show u(wrk, zlist_m)

    (u(wrk, zlist_p) - u(wrk, zlist_m)) / 2ϵ
end;

function du(wrk, zlist; α=1, acc=false, cal_val=false)
    for i = 1:length(zlist)-1
        if !acc
            wrk.grad[i] = 0
        end
        wrk.grad[i] -= 2α * (zlist[i+1] - zlist[i]) / length(zlist) #/ wrk.h^2
        wrk.grad[i+1] += 2α * (zlist[i+1] - zlist[i]) / length(zlist) #/ wrk.h^2
    end
    wrk.grad[1] = 0
    wrk.grad[length(zlist)] = 0

    if cal_val
        return u(wrk, zlist; α=α)
    end
end


using LBFGSB
using Printf

function custom_optimize(f, df, zlist_guess; attempts=1, maxiter=15000, pgtol=1e-7, α=1)

    optimizer = L_BFGS_B(length(zlist_guess), 17)

    # set up bounds
    bounds = zeros(3, length(zlist))
    for i = 1:length(zlist)
        bounds[1, i] = 0
        bounds[2, i] = -10μm
        bounds[3, i] = 60μm
    end

    fout_best = f(zlist_guess)
    zout_best = copy(zlist_guess)

    @printf("%-10.9f\n", fout_best)

    for i = 1:attempts
        fout, zout = optimizer(
            f,
            df,
            zlist_guess,
            bounds,
            m=5,
            factr=1e7,
            pgtol=pgtol,
            iprint=-1,
            maxfun=15000,
            maxiter=maxiter
        )

        if fout < fout_best
            fout_best = fout
            zout_best = copy(zout)
        end

        @printf("%-10.5f\n", fout_best)
    end

    zout_best
end
