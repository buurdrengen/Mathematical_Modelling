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

K = [[300 140 40], [ 500 230 60], [1000 400 70 ]]


df = Array(CSV.read("Depression/channel_data_interp.csv", DataFrame, header = 0))

function compute_Cbc(n_cores, H, lastbomb)
    h = length(H)

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
    @variable(myModel, y[1:h-1], Bin)


    @objective(myModel, Min, sum(t))

    @constraint(myModel, y .== x0[1:h-1] .+ x0[2:h] .+ x1[1:h-1] .+ x1[2:h] .+ x2[1:h-1] .+ x2[2:h])
    @constraint(myModel, x0 .+ x1 .+ x2 .<= 1)
    @constraint(myModel, t .>= R .- B)
    @constraint(myModel, t .>= B .- R)
    @constraint(myModel, R .>= B )
    @constraint(myModel, [i=1:h],R[i] == sum(A0[i,j]*x0[j] .+ A1[i,j]*x1[j] .+ A2[i,j]*x2[j] for j=1:h))
    @constraint(myModel, x0[1] <= 1 - lastbomb)
    @constraint(myModel, x1[1] <= 1 - lastbomb)
    @constraint(myModel, x2[1] <= 1 - lastbomb)

    # set_optimizer_attribute(myModel, "maxNodes", 1000000)
    set_optimizer_attribute(myModel, "threads", n_cores)
    set_optimizer_attribute(myModel, "logLevel", 0)

    optimize!(myModel)

    if termination_status(myModel) == MOI.OPTIMAL || termination_status(myModel) == MOI.NODE_LIMIT
        println(" Done!")
    else
        println("Optimize was not succesful. Return code: ", termination_status(myModel))
    end

    tt = JuMP.solve_time(myModel)
    objective = JuMP.objective_value(myModel)
    nbombs = sum(JuMP.value.(x0) .+ JuMP.value.(x1) .+ JuMP.value.(x2))

    vx0 = JuMP.value.(x0)
    vx1 = JuMP.value.(x1)
    vx2 = JuMP.value.(x2)

    Ri = JuMP.value.(R)
    lastbomb = vx0[end] + vx1[end] + vx2[end]
    carry1 = A0[end,end-1]*vx0[end] + A1[end,end-1]*vx1[end] + A2[end,end-1]*vx2[end] + A0[end,end-2]*vx0[end-1] .+ A1[end,end-2]*vx1[end-1] .+ A2[end,end-2]*vx2[end-1]
    carry2 = A0[end,end-2]*vx0[end] + A1[end,end-2]*vx1[end] + A2[end,end-2]*vx2[end]
    carry = [carry1, carry2]
    println(carry)

    return tt, objective, nbombs, Ri, lastbomb, carry
end

m = length(df[:,1])
scale = 2
nn = round.(Int,LinRange(1,m,scale+1))

n_cores = 24

total_time_cbc_16 = zeros(scale)
objective_value = zeros(scale)
number_bombs = zeros(scale)

HR = df[:,2]
RR = zeros(m)


# Preload
# compute_Cbc(1,1)

for i in 1:scale
    if i ==1
        global lb = 0
    end
    n1 = nn[i]
    n2 = nn[i+1]
    nH = HR[n1:n2]
    print("Cbc16,i = ", i)
    t, obj, nb, Ri, lb2, carry = compute_Cbc(n_cores, nH, lb)
    RR[n1:n2] = Ri
    HR[n1:n2] = HR[n1:n2] - Ri
    if i < scale
        HR[n2+1:n2+2] = HR[n2+1:n2+2] .- carry
    end
    total_time_cbc_16[i] = t
    objective_value[i] = obj
    number_bombs[i] = nb
    global lb = lb2
end

# println(total_time)
# println(objective_value)
# println(number_bombs)

plt = plot((1:m)*0.25 ,HR, xlabel = "Distance from Ocean [km]", ylabel = "Height above sea level [m]", title = "Depth of Channel", label = "Channel", markercolor = :red, markershape = :circle, markersize = 2)
plot!(xy,max_h, linecolor = "black", label = "Minimum Depth")
# plot!((1:m)*scale,total_time_cbc_1, lw = 2, label = "Cbc single thread")
# plot!((1:m)*scale,total_time_glpk, lw = 2, label = "GLPK")

savefig(plt,"Depression/figures/Opgave6e.png")
println("Objective Value = ", sum(objective_value))
println("Number of Nukes = ", sum(number_bombs))
println("Computation Time = ", sum(total_time_cbc_16))

