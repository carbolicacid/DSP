#Compile and run the codes used for plotting and sub-plotting different signals and show the output waveforms.

import numpy as np
import matplotlib.pyplot as plt
t = np.linspace(0,1, 100)
A = 1 
f = 5
phi = [0,90,180,270]

for i in range(len(phi)):
    x = A*np.sin(2*np.pi*f*t + phi[i]*np.pi/180)
    plt.subplot(2,2,i+1)
    plt.plot(t,x, 'r')
    plt.xlabel('Time(t)')
    plt.ylabel('Amplitude')
    plt.title(f'Phase {phi[i]}')
    plt.grid()
    plt.tight_layout()
    
    
 
    
 
    
import numpy as np
import matplotlib.pyplot as plt
t = np.linspace(0,1,100)
A = 1 
f = 5
phi1,phi2,phi3 = 0,120,240
x1 = A*np.sin(2*np.pi*f*t + phi1*np.pi/180)
x2 = A*np.sin(2*np.pi*f*t + phi2*np.pi/180)
x3 = A*np.sin(2*np.pi*f*t + phi3*np.pi/180)
plt.plot(t,x1, 'r')
plt.plot(t,x2, 'g')
plt.plot(t,x3, 'b')
plt.xlabel('Time(t)')
plt.ylabel('Amplitude')
plt.title('three phase sinusoidal signal')
plt.legend(['Phase-1', 'Phase-2', 'Phase-3'], loc = 1)
plt.grid()
plt.tight_layout()


    
    
#Generate the following sinusoidal signal x(t) = A sin(2πf t + ϕ) with the amplitude A = 2V , frequency f = 10Hz and phase ϕ = 0. Let the length of the signal be 3 periods. Plot the signal.

import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,0.3, 100)
A = 2
f = 10
phi = 0

x1 = A*np.sin(2*np.pi*f*t + phi*np.pi/180)
plt.plot(t,x1)
plt.xlabel('Time(t)', plt.ylabel('Amplitude'))
plt.title('Sinusoidal Signal')


#Generate a sinusoidal signal with 10kHz and show it for 10ms. Assume amplitude values and necessary data. Use the sin() function first. Then use the cos() function to repeat the same.

import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,0.001, 100)
A = 5
f = 10000
phi = 0

plt.subplot(121)
x1 = A*np.sin(2*np.pi*f*t + phi*np.pi/180)
plt.plot(t, x1, 'r')
plt.xlabel('Time(t)')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(122)
x2 = A*np.cos(2*np.pi*f*t + phi*np.pi/180)
plt.plot(t, x2, 'y')
plt.xlabel('Time(t)')
plt.ylabel('Amplitude')
plt.grid()
plt.tight_layout()






#Solve all the exercise problems of the textbook from page 31.

import numpy as np
import matplotlib.pyplot as plt
















