using PEtab
using Aqua

@testset "Aqua" begin
    Aqua.test_ambiguities(PEtab, recursive = false)
    Aqua.test_undefined_exports(PEtab)
    Aqua.test_unbound_args(PEtab) 
    Aqua.test_stale_deps(PEtab)
    Aqua.test_deps_compat(PEtab)
    Aqua.test_piracies(PEtab)
    Aqua.test_project_extras(PEtab)
    Aqua.find_persistent_tasks_deps(PEtab)
end
