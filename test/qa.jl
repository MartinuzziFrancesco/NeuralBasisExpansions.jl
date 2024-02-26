using Aqua: Aqua
#using JET: JET
using NeuralBasisExpansions
using JuliaFormatter

@test JuliaFormatter.format(NeuralBasisExpansions; verbose=false, overwrite=false)
Aqua.test_all(NeuralBasisExpansions; ambiguities=false, deps_compat=(check_extras = false))
#JET.test_package(NBeats; target_defined_modules=true)
