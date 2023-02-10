import numpy as np
import time

def tensor_solve(Vx,Vy,Vt, N = 3):
    """
    Vx, Vy, and Vt must all be the same shape numpy array
    """
    # Assume N = 3

    if N%2 == 0: Warning("N must be odd!")
    r = (N-1)//2
    (n,m) = np.shape(Vx)

    si = (n-2*r)*(m-2*r); sj = N**2  # Matrix dimensions

    A0 = np.zeros((si,sj,2))
    b0 = np.zeros((si,sj,1))
    vector_field = np.zeros((n,m,2))

    for i in range(sj):
        x = i%N; y = i//N
        u = x-(r+1); v = y-(r+1)
        if u == 0: u = None
        if v == 0: v = None

        A0[:,i,0] = Vx[x:u, y:v].flatten()
        A0[:,i,1] = Vy[x:u, y:v].flatten()
        b0[:,i,0] = Vt[x:u, y:v].flatten()

    AT = np.transpose(A0,(0,2,1))
    A = np.matmul(AT,A0)
    b = np.matmul(AT,b0)

    sol = np.linalg.solve(A,-b)
    vector_field[r:-r,r:-r,:] = np.reshape(sol,(n-2*r,m-2*r,2))
    vector_field = np.transpose(vector_field,(2,0,1))

    return vector_field


if __name__ == "__main__":
    print("This is a tensor solver")

    print("Testing Linalg Solve ..")

    sample_size = (480,640)
    result = np.zeros(1000)

    Vx = np.random.rand(sample_size[0],sample_size[1])
    Vy = np.random.rand(sample_size[0],sample_size[1])
    Vt = np.random.rand(sample_size[0],sample_size[1])

    N = 5


    for i in range(1000):
        if i%10 == 0: print(f"Completion: {i//10}%  ", end = "\r")
        Vx[:,:] = np.random.rand(sample_size[0],sample_size[1])
        Vy[:,:] = np.random.rand(sample_size[0],sample_size[1])
        Vt[:,:] = np.random.rand(sample_size[0],sample_size[1])

        start = time.time()
        output = tensor_solve(Vx, Vy, Vt, N = 3)
        result[i] = time.time() - start

    print("\nDone!\n")

    mu = np.mean(result)
    sigma = np.std(result)
    print(f"Average is {np.round(mu*1000,1)}ms and std is {np.round(sigma*1000,1)}ms")

    # pos = np.mgrid[0:6,0:6]
    # vector_field = np.zeros((2,6,6))
    # x_list = pos[0].flatten()
    # y_list = pos[1].flatten()

    # r = 1


    # for j in range(np.size(x_list)):
    #     # Try to implement np.tensordot
    #     x0 = x_list[j]; y0 = y_list[j]
    #     u1 = x0-r; u2 = x0+r+1; 
    #     v1 = y0-r; v2 = y0+r+1
    #     if u1 == 0: u1 = None
    #     if v1 == 0: v1 = None

    #     Vx_p = Vx[v1:v2, u1:u2].flatten()
    #     Vy_p = Vy[v1:v2, u1:u2].flatten()
    #     Vt_p = Vt[v1:v2, u1:u2].flatten()

    #     A = np.stack((Vx_p,Vy_p))

    #     sol = np.linalg.lstsq(A.T, -Vt_p, rcond=None)
    #     vector_field[0, x0, y0] = sol[0][0]
    #     vector_field[1, x0, y0] = sol[0][0]



    # vector_field_tensor = tensor_solve(Vx,Vy,Vt)

    # print(vector_field[:,1:-1,1:-1])
    # print(vector_field_tensor[1:-1,1:-1,:])