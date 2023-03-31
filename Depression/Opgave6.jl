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

# K = [
# 300 140 40
# ]

K = [[300 140 40], [ 500 230 60], [1000 400 70 ]]


n = 200

df = Array(CSV.read("Depression/channel_data_interp.csv", DataFrame, header = 0))
m = length(df[:,1])
xy = df[n:m,1]
H = df[n:m,2]
h = length(H)
println("h = ",h)

A0, B = constructA(H,K[1])
A1, _ = constructA(H,K[2])
A2, _ = constructA(H,K[3])

myModel = Model(Cbc.Optimizer)
    # If your want ot use GLPK instead use:
# myModel = Model(GLPK.Optimizer)

@variable(myModel, x0[1:h], Bin )
@variable(myModel, x1[1:h], Bin )
@variable(myModel, x2[1:h], Bin )
@variable(myModel, R[1:h] >= 0 )
@variable(myModel, t[1:h] >= 0)
# @variable(myModel, y[1:h-1], Bin)


@objective(myModel, Min, sum(t))

@constraint(myModel, 1 .>= x0[1:h-1] .+ x0[2:h] .+ x1[1:h-1] .+ x1[2:h] + x2[1:h-1] .+ x2[2:h])
@constraint(myModel, x0 .+ x1 .+ x2 .<= 1)
@constraint(myModel, t .>= R .- B)
@constraint(myModel, t .>= B .- R)
@constraint(myModel, R .>= B )
@constraint(myModel, [i=1:h],R[i] == sum(A0[i,j]*x0[j] .+ A1[i,j]*x1[j] + A2[i,j]*x2[j] for j=1:h))

    set_optimizer_attribute(myModel, "seconds", 3600)
# set_optimizer_attribute(myModel, "maxNodes", 100000)
set_optimizer_attribute(myModel, "threads", 24)

optimize!(myModel)

if termination_status(myModel) == MOI.OPTIMAL || termination_status(myModel) == MOI.NODE_LIMIT || termination_status(myModel) == MOI.TIME_LIMIT
    println("Objective value: ", JuMP.objective_value(myModel))
else
    println("Optimize was not succesful. Return code: ", termination_status(myModel))
end

nR =  JuMP.value.(R)
nx0 = JuMP.value.(x0)
nx1 = JuMP.value.(x1)
nx2 = JuMP.value.(x2)
sx = sum(nx0 .+ nx1 .+ nx2)

open("summary.txt", "w") do f
    print(f,solution_summary(myModel))
end

println("Low Yield:", sum(nx0))
println("Medium Yield:", sum(nx1))
println("High Yield:", sum(nx2))
println(JuMP.solve_time(myModel))

new_H = H .- nR

max_h = -10*ones(h)



plt = plot(xy, new_H, title = "Depth of Channel", label = "Channel", ylabel = "Depth [m]", xlabel = "Distance from Ocean [km]", legend = :bottom)
plot!(xy,max_h, linecolor = "black", label = "Minimum Depth")
# savefig(plt,"Depression/figures/Opgave6.png")
