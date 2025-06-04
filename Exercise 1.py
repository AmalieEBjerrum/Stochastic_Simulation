import matplotlib.pyplot as plt

#Exercise 1: LCG

#Defining the LCG as a function
def LCG(x0, M, a, c, numberofrandomnumbers):
    #Initialise the seed state and define list for storing random numbers
    x = [x0]

    #Generate required numbers of random numbers
    for i in range (1,numberofrandomnumbers):
        x.append(x[i])

    return x
    

#Defining the parameter values
numberofrandomnumbers = 10000
x0 = 1
M = 16
a = 5
c = 1

#Generate random numbers by calling the function
randomnumbers = LCG(x0, M, a, c, numberofrandomnumbers)


#Plotting histogram with 10 classes
plt.hist(randomnumbers, bins=10, edgecolor='black')
plt.title('Histogram of LCG Random Numbers')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()







