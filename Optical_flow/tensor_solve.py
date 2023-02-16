import numpy as np
import time

def tensor_solve(Vx, Vy, Vt, N = 3):
    """
    Vx, Vy, and Vt must all be the same shape numpy array
    """
    # Assume N = 3

    (n,m) = np.shape(Vx)
    vector_field = np.zeros((2,n,m))

    if (np.shape(Vy) != (n,m)) or (np.shape(Vt) != (n,m)): raise Exception("Gradients Must be the same shape!")
    if N%2 == 0: raise Exception("N must be odd!")

    # Somehow this method only works for square matrices
    # Padding is therefore nessersary
    k = max(n,m)
    Vx = np.pad(Vx,((0,k-n),(0,k-m)), mode="constant")
    Vy = np.pad(Vy,((0,k-n),(0,k-m)), mode="constant")
    Vt = np.pad(Vt,((0,k-n),(0,k-m)), mode="constant")

    # Matrix dimensions
    r = (N-1)//2; d = 2*r
    si = (k-d)**2; sj = N**2  

    # Predefinitions
    A0 = np.zeros((si,sj,2))
    b0 = np.zeros((si,sj,1))

    # Grab all proper submatrices of size (k-d) by (k-d)
    for i in range(sj):
        x = i%N; y = i//N
        u = x-d; v = y-d
        if u == 0: u = None
        if v == 0: v = None

        A0[:,i,0] = Vx[y:v,x:u].T.flatten()
        A0[:,i,1] = Vy[y:v,x:u].T.flatten()
        b0[:,i,0] = Vt[y:v,x:u].T.flatten()

    
    # All matricies in A must be square. This is done by 3D matrix multiplication
    AT = np.transpose(A0, (0,2,1))
    A = np.matmul(AT, A0)
    b = np.matmul(AT, b0)

    # Make sure trivial zeros does not kill the solver :)
    # 'A' cannot be singular, so this is fixed here
    trivial_zeros = np.argwhere(np.all(A[..., :] == 0, axis=(1,2)))
    A[trivial_zeros] = np.array([[1,1],[0,1]])
    # Magic!
    try:
        sol = np.linalg.solve(A,-b)
    except np.linalg.LinAlgError:
        sol = np.zeros((si,2,1))

    # Reconstruct the vector field to the original size
    vector_field[:,r:-r,r:-r] = np.reshape(sol,(k-d,k-d,2))[:n-d,:m-d].transpose((2,0,1))
    
    return vector_field


#-------------------------------------------------------------------

# def square_tensor_solve(Vx, Vy, Vt, N = 3):
#     """
#     Vx, Vy, and Vt must all be the same shape square numpy array
#     """
#     # Assume N = 3

#     (n,m) = np.shape(Vx)

#     vector_field = np.zeros((2,n,m))

#     if not n == m: raise Exception("Gradients must be square! - Use ordinary tensor solve otherwise")
#     if (np.shape(Vy) != (n,m)) or (np.shape(Vt) != (n,m)): raise Exception("Gradients Must be the same shape!")
#     if N%2 == 0: raise Exception("N must be odd!")

#     # No padding nessersary :)
#     k = n
#     r = (N-1)//2; d = 2*r
#     si = (k-d)**2; sj = N**2  # Matrix dimensions

#     A0 = np.zeros((si,sj,2))
#     b0 = np.zeros((si,sj,1))

#     # grab all submatrices of size (k-d) by (k-d)
#     for i in range(sj):
#         x = i%N; y = i//N
#         u = x-d; v = y-d
#         if u == 0: u = None
#         if v == 0: v = None

#         A0[:,i,0] = Vx[y:v,x:u].T.flatten()
#         A0[:,i,1] = Vy[y:v,x:u].T.flatten()
#         b0[:,i,0] = Vt[y:v,x:u].T.flatten()

#     AT = np.transpose(A0,(0,2,1))
#     A = np.matmul(AT, A0)
#     b = np.matmul(AT, b0)

#     # Magic!
#     sol = np.linalg.solve(A,-b)

#     # Reconstruct the vector field to the original size
#     vector_field[:,r:-r,r:-r] = np.reshape(sol,(k-d,k-d,2)).transpose((2,0,1))

#     return vector_field


#-------------------------------------------------------------------


if __name__ == "__main__":

    print("Testing Linalg Solve ..")

    sample_size = (11,125)
    n_samples = 1
    N = 7

    r = (N-1)//2
    if n_samples >= 10:
        n_percent = n_samples/100
    else:
        n_percent = n_samples

    result1 = np.zeros(n_samples); result2 = np.copy(result1)

    Vx = np.random.rand(sample_size[0],sample_size[1])
    Vy = np.random.rand(sample_size[0],sample_size[1])
    Vt = np.random.rand(sample_size[0],sample_size[1])


    for i in range(n_samples):
        if i%n_percent == 0: print(f"Completion: {i//n_percent}%  ", end = "\r")
        Vx[:,:] = np.random.rand(sample_size[0],sample_size[1])
        Vy[:,:] = np.random.rand(sample_size[0],sample_size[1])
        Vt[:,:] = np.random.rand(sample_size[0],sample_size[1])

        start = time.time()
        output1 = tensor_solve(Vx, Vy, Vt, N = N)
        result1[i] = time.time() - start

    print("\nDone!\n")

    mu1 = np.mean(result1)
    sigma1 = np.std(result1)

    print("Testing iterative lstsq loop ..")

    pos = np.mgrid[0:sample_size[0],0:sample_size[1]]
    vector_field = np.zeros((2,sample_size[0],sample_size[1]))
    x_list = pos[0].flatten()
    y_list = pos[1].flatten()

    for i in range(n_samples):
        if i%n_percent == 0: print(f"Completion: {i//n_percent}%  ", end = "\r")
        Vx[:,:] = np.random.rand(sample_size[0],sample_size[1])
        Vy[:,:] = np.random.rand(sample_size[0],sample_size[1])
        Vt[:,:] = np.random.rand(sample_size[0],sample_size[1])
        
        start = time.time()

        for j in range(np.size(x_list)):
            # Try to implement np.tensordot
            x0 = x_list[j]; y0 = y_list[j]
            u1 = x0-r; u2 = x0+r+1; 
            v1 = y0-r; v2 = y0+r+1
            if u1 == 0: u1 = None
            if v1 == 0: v1 = None

            Vx_p = Vx[v1:v2, u1:u2].flatten()
            Vy_p = Vy[v1:v2, u1:u2].flatten()
            Vt_p = Vt[v1:v2, u1:u2].flatten()

            A = np.stack((Vx_p,Vy_p))

            sol = np.linalg.lstsq(A.T, -Vt_p, rcond=None)
            vector_field[0, x0, y0] = sol[0][0]
            vector_field[1, x0, y0] = sol[0][1]

        output2 = vector_field

        result2[i] = time.time() - start

    print("\nDone!\n")
    mu2 = np.mean(result2)
    sigma2 = np.std(result2)


    print("Comparing...")

    output1 = tensor_solve(Vx, Vy, Vt, N = N)

    # print(np.shape(output1))
    # print(np.shape(output2))

    # print(output1[:,r:-r,r:-r])
    # print(30*"-")
    # print(output2[:,r:-r,r:-r])

    working_precision = 6
    n = 0
    nn = 0

    for k in zip(output1[:,r:-r,r:-r].flatten(), output2[:,r:-r,r:-r].flatten()):
        if np.round(k[0], working_precision) != np.round(k[1], working_precision):
            if k[1]: nn += 1; print(f"Error! {np.round(k[0], working_precision)} != {np.round(k[1], working_precision)}")
            n += 1

    if n == 0:
        print("Results compare OK!\n")
    else:
        print(f"Found {n} errors of which {nn} is nontrivial!\n")

    print(f"Tensor Solve: Average is {np.round(mu1*1000,1)}ms and std is {np.round(sigma1*1000,1)}ms")
    print(f"Lstsq Loop: Average is {np.round(mu2*1000,1)}ms and std is {np.round(sigma2*1000,1)}ms\n")

    if mu1 > mu2:
        print(f"Lstsq Loop is {np.round((mu1/mu2 - 1)*100,1)} percent faster")
    else:
        print(f"Tensor Solve is {np.round((mu2/mu1 - 1)*100,1)} percent faster")