import numpy as np
import time

def tensor_solve(Vx,Vy,Vt, N = 3):
    """
    Vx, Vy, and Vt must all be the same shape numpy array
    """
    # Assume N = 3

    r = (N-1)//2; d = 2*r
    (n,m) = np.shape(Vx)

    si = (n-d)*(m-d); sj = N**2  # Matrix dimensions

    A0 = np.zeros((si,sj,2))
    b0 = np.zeros((si,sj,1))
    vector_field = np.zeros((n,m,2))

    if N%2 == 0: Warning("N must be odd!"); return np.transpose(vector_field,(2,1,0))

    for i in range(sj):
        x = i%N; y = i//N
        u = x-d; v = y-d
        if u == 0: u = None
        if v == 0: v = None

        A0[:,i,0] = Vx[x:u, y:v].flatten()
        A0[:,i,1] = Vy[x:u, y:v].flatten()
        b0[:,i,0] = Vt[x:u, y:v].flatten()

    AT = np.transpose(A0,(0,2,1))
    A = np.matmul(AT,A0)
    b = np.matmul(AT,b0)

    sol = np.linalg.solve(A,-b)

    vector_field[r:-r,r:-r,:] = np.reshape(sol,(n-d,m-d,2))
    vector_field = np.transpose(vector_field,(2,1,0))

    return vector_field


#-------------------------------------------------------------------


if __name__ == "__main__":

    print("Testing Linalg Solve ..")

    sample_size = (100,100)
    result1 = np.zeros(1000); result2 = np.copy(result1)

    Vx = np.random.rand(sample_size[0],sample_size[1])
    Vy = np.random.rand(sample_size[0],sample_size[1])
    Vt = np.random.rand(sample_size[0],sample_size[1])

    N = 5
    r = (N-1)//2


    for i in range(1000):
        if i%10 == 0: print(f"Completion: {i//10}%  ", end = "\r")
        Vx[:,:] = np.random.rand(sample_size[0],sample_size[1])
        Vy[:,:] = np.random.rand(sample_size[0],sample_size[1])
        Vt[:,:] = np.random.rand(sample_size[0],sample_size[1])

        start = time.time()
        output1 = tensor_solve(Vx, Vy, Vt, N = N)
        result1[i] = time.time() - start

    print("\nDone!\n")

    print("Testing iterative lstsq loop ..")

    mu1 = np.mean(result1)
    sigma1 = np.std(result1)

    pos = np.mgrid[0:sample_size[0],0:sample_size[1]]
    vector_field = np.zeros((2,sample_size[0],sample_size[1]))
    x_list = pos[0].flatten()
    y_list = pos[1].flatten()

    for i in range(1000):
        if i%10 == 0: print(f"Completion: {i//10}%  ", end = "\r")
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
    working_precision = 6
    n = 0

    for k in zip(output1[:,r:-r,r:-r].flatten(), output2[:,r:-r,r:-r].flatten()):
        if np.round(k[0], working_precision) != np.round(k[1], working_precision):
            print(f"Error! {np.round(k[0], working_precision)} != {np.round(k[1], working_precision)}")
            n += 1

    if n == 0:
        print("Results compare OK!\n")
    else:
        print(f"Found {n} errors!\n")

    print(f"Tensor Solve: Average is {np.round(mu1*1000,1)}ms and std is {np.round(sigma1*1000,1)}ms")
    print(f"Lstsq Loop: Average is {np.round(mu2*1000,1)}ms and std is {np.round(sigma2*1000,1)}ms\n")

    if mu1 > mu2:
        print(f"Lstsq Loop is {np.round((mu1/mu2 - 1)*100,1)} percent faster")
    else:
        print(f"Tensor Solve is {np.round((mu2/mu1 - 1)*100,1)} percent faster")