# Txt tests in julia # 

using GLPK, Cbc, JuMP, SparseArrays
using CSV
using DataFrames
using DelimitedFiles
using Plots



function constructA(H,K)
    h = length(H)
    k = length(K)-1
    A = zeros(h,h+2*k)

    for i in 1:h
        A[i,i:i+2*k] = hcat(transpose(reverse(K[2:end])), K)
    end
    
    A = A[:,k+1:end-k]

    B = H .+ 10

    # Make a function that returns A when given H and K
    return A, B
end

K = [
300 140 40
]

K = [[300 140 40], [ 500 230 60]]

df = Array(CSV.read("Depression/channel_data_interp.csv", DataFrame, header = 0))
xy = df[:,1]
H = df[:,2]
h = length(H)

A0, B = constructA(H,K[1])
A1, _ = constructA(H,K[2])

myModel = Model(Cbc.Optimizer)
    # If your want ot use GLPK instead use:
# myModel = Model(GLPK.Optimizer)

@variable(myModel, x0[1:h], Bin )
@variable(myModel, x1[1:h], Bin )
@variable(myModel, R[1:h] >= 0 )
@variable(myModel, t[1:h] >= 0)
@variable(myModel, y[1:h-1], Bin)


@objective(myModel, Min, sum(t))

@constraint(myModel, y .== x0[1:h-1] .+ x0[2:h] .+ x1[1:h-1] .+ x1[2:h])
# @constraint(myModel, x0 .+ x1 .<= 1)
@constraint(myModel, t .>= R .- B)
@constraint(myModel, t .>= B .- R)
@constraint(myModel, R .>= B )
@constraint(myModel, [i=1:h],R[i] == sum(A0[i,j]*x0[j] .+ A1[i,j]*x1[j] for j=1:h))

# set_optimizer_attribute(myModel, "maxNodes", 1000000)
set_optimizer_attribute(myModel, "threads", 16)

JuMP.optimize!(myModel)

if termination_status(myModel) == MOI.OPTIMAL || termination_status(myModel) == MOI.NODE_LIMIT
    println("Objective value: ", JuMP.objective_value(myModel))
else
    println("Optimize was not succesful. Return code: ", termination_status(myModel))
end

nR =  JuMP.value.(R)
nx0 = JuMP.value.(x0)
nx1 = JuMP.value.(x1)
sx = sum(nx0 .+ nx1)

tt = JuMP.solve_time(myModel)

new_H = H .- nR

max_h = -10*ones(h)



plt = plot(xy, new_H, title = "Depth of Channel", label = "Channel", ylabel = "Depth [m]", xlabel = "Distance from Ocean [km]", legend = :bottom, markercolor = :red, markershape = :circle, markersize = 2)
plot!(xy,max_h, linecolor = "black", label = "Minimum Depth")
savefig(plt,"Depression/figures/Opgave6d.png")