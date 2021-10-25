using LinearAlgebra

function rand_pd_mat(T, n)
    U = qr(randn(T, n, n)).Q
    return Matrix(Symmetric(U * rand_pd_diag_mat(T, n) * U'))
end

rand_pd_diag_mat(T, n) = Diagonal(rand(T, n))
