using TestItemRunner

# @testitem "Time steppers" begin include("timesteppers.jl") end

# Only run tests from this test dir, and not from other packages in monorepo

function myfilter(t)
    #return occursin(@__DIR__, t.filename)
    return endswith(t.filename,"operators_io.jl")
end

@run_package_tests filter = myfilter