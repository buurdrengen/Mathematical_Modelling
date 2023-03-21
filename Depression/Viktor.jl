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

df = Array(CSV.read("Depression/channel_data_interp.csv", DataFrame, header = 0))
xy = df[:,1]
H = df[:,2]
h = length(H)

A, B = constructA(H,K)

myModel = Model(Cbc.Optimizer)
    # If your want ot use GLPK instead use:
    #myModel = Model(GLPK.Optimizer)

@variable(myModel, x[1:h], Bin )
@variable(myModel, R[1:h] >= 0 )

@objective(myModel, Min, abs(R .- B) )

@constraint(myModel, [j=1:h],R[j] >= H[j] + 10 )
@constraint(myModel, [i=1:h],R[i] == sum(A[i,j]*x[j] for j=1:h) )

optimize!(myModel)

if termination_status(myModel) == MOI.OPTIMAL
    println("Objective value: ", JuMP.objective_value(myModel))
    println("x = ", JuMP.value.(x))
    println("R = ", JuMP.value.(R))
else
    println("Optimize was not succesful. Return code: ", termination_status(myModel))
end

nR =  JuMP.value.(R)

new_H = H .- nR

max_h = -10*ones(h)

plt = plot(xy,new_H, title = "Depth of Channel", label = "Channel", ylabel = "Depth [m]", xlabel = "Distance from Ocean [km]", legend = :bottom)
plot!(xy,max_h, linecolor = "black", label = "Minimum Depth")
savefig(plt,"Opgave2.png")