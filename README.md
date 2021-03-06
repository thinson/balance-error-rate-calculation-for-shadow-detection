# balance-error-rate-calculation-for-shadow-detection
This is a repository for balance error rate calculation.
# about BER(balance error rate)
Since Np is usually much smaller than Nn in natural images, we employ the second metric called the balance error rate (BER) to obtain a more balanced evaluation by equally considering the shadow and non-shadow regions:

![ber](https://latex.codecogs.com/gif.latex?BER=\left(%201-\frac{1}{2}\left(%20\frac{TP}{N_p}+\frac{TN}{N_n}%20\right)%20\right))

Note that unlike the accuracy metric, for BER, the lower its value, the better the detection result is.

