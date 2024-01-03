using NBeats
using Documenter

DocMeta.setdocmeta!(NBeats, :DocTestSetup, :(using NBeats); recursive=true)

makedocs(;
    modules=[NBeats],
    authors="Francesco Martinuzzi",
    repo="https://github.com/MartinuzziFrancesco/NBeats.jl/blob/{commit}{path}#{line}",
    sitename="NBeats.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://MartinuzziFrancesco.github.io/NBeats.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/MartinuzziFrancesco/NBeats.jl", devbranch="main")
