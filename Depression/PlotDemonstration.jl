using PlotlyBase, PlotlyJS

raw_symbols = PlotlyBase.get_plotschema().traces[:scatter][:attributes][:marker][:symbol][:values]
namestems = []
namevariants = []
symbols = []
for i in 1:3:length(raw_symbols)
    name = raw_symbols[i+2]
    namestem = replace(replace(name, "-open" => ""), "-dot" => "")
    push!(symbols, raw_symbols[i])
    push!(namestems, namestem)
    push!(namevariants, name[(length(namestem)+1):end])
end



trace = scatter(
    x=namevariants, y=namestems, mode="markers",
    marker=attr(
        symbol=raw_symbols[218],
        line=attr(width=2, color="black"),
        color="yellow", size=15
    ),
    hovertemplate="name: %{y}%{x}<br>number: %{marker.symbol}<extra></extra>"
)
layout = Layout(
    title="Mouse over symbols for name & number!",
    xaxis_range=[-1,4], yaxis_range=[length(unique(namestems)),-1],
    margin=attr(b=0,r=0), xaxis_side="top", height=1400, width=400
)
plot([trace], layout)