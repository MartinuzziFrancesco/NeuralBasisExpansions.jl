using Aqua: Aqua
#using JET: JET
using NBeats
using JuliaFormatter

@test JuliaFormatter.format(NBeats; verbose=false, overwrite=false)
Aqua.test_all(NBeats; ambiguities=false, deps_compat=(check_extras = false))
#JET.test_package(NBeats; target_defined_modules=true)
