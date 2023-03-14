# Txt tests in julia # 
using DelimitedFiles
f = readdlm("Depression/channel_data.txt",'\t','\n')
println(f)
close(f)