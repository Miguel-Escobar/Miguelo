import numpy as np
import matplotlib.pyplot as plt

N=64
M=64
delta = int(M/20)
deltax=1/N
deltat=1/M

e_pot = np.empty(int(2*np.pi*M))
e_cin = np.empty(int(2*np.pi*M))
tiempo = np.empty(int(2*np.pi*M))
v = np.empty(N)
e = np.empty(N)
n = np.empty(N)
x = np.empty(N)
k2 = np.empty(N)
fft_v = np.empty(N, dtype=complex)

n0=0.01
for i in range(N):
    n[i] = n0*np.sin(2*np.pi*i/N) # densidad de carga
    v[i] = 0. # vel de elecs

# primer medio paso
fft_n = np.fft.fft(n)
num_onda = np.fft.fftfreq(fft_n.shape[0],d=1)
for i in range(N):
    x[i] = i*deltax
    k2[i] = 4.*np.sin(np.pi*num_onda[i])**2
    if k2[i] > 0:
        fft_v[i] = -fft_n[i]*deltax**2/k2[i]
    else:
        fft_v[i] = 0.
u = np.real(np.fft.ifft(fft_v))    
for i in range(N):
    if i < N-1 and i > 0:
        v[i] = v[i] + .25*(deltat/deltax)*(u[i+1]-u[i-1])
    else:
        if i==N-1:
            v[i] = v[i] + .25*(deltat/deltax)*(u[0]-u[i-1])
        else:
            v[i] = v[i] + .25*(deltat/deltax)*(u[i+1]-u[N-1])

# los siguientes pasos:
count = 0
for j in range(int(2*np.pi*M)):
    for i in range(N):
        if i > 0 and i < N-1:
            n[i] = n[i] + .5*(deltat/deltax)*(v[i+1]-v[i-1])
        else:
            if i==0:
                n[i] = n[i] + .5*(deltat/deltax)*(v[i+1]-v[N-1])
            else:
                n[i] = n[i] + .5*(deltat/deltax)*(v[0]-v[i-1])
    
    fft_n = np.fft.fft(n)
    for i in range(N):
        if k2[i] > 0:
            fft_v[i] = fft_n[i]*deltax**2/k2[i]
        else:
            fft_v[i] = 0.
    u = np.real(np.fft.ifft(fft_v)) 
    for i in range(N):
        if i < N-1 and i > 0:
            v[i] = v[i] + .5*(deltat/deltax)*(u[i+1]-u[i-1])
            e[i] = .5*(u[i+1]-u[i-1])/deltax
        else:
            if i==N-1:
                v[i] = v[i] + .5*(deltat/deltax)*(u[0]-u[i-1])
                e[i] = .5*(u[0]-u[i-1])/deltax
            else:
                v[i] = v[i] + .5*(deltat/deltax)*(u[i+1]-u[N-1])
                e[i] = .5*(u[i+1]-u[N-1])/deltax

    e_pot[j] = np.mean(e**2/2)
    e_cin[j] = np.mean(v**2/2)
    tiempo[j] = j*deltat/(2*np.pi)
        
    if j % delta == 0:
        plt.figure(figsize=(10,10))
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.ylim(-.002,.002)
        plt.plot(x,v,color='blue')

        print('count=',count)
        count = count +1
        plt.close()
plt.figure(figsize=(10,10))
plt.xlabel('$t$')
plt.ylabel('$energia$')

plt.plot(tiempo,e_pot,color='red', label='Energia Pot')
plt.plot(tiempo,e_cin,color='blue', label='Energia Cin')
plt.plot(tiempo,e_cin+e_pot,color='black', label='Energia Tot')
plt.legend(loc='upper right', frameon=False)
plt.ylim(0,7.5e-7)
plt.savefig('energia', dpi=None)
plt.close()