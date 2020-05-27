# balance-error-rate-calculation-for-shadow-detection
This is a repository for balance error rate calculation.
# about BER(balance error rate)
Since Np is usually much smaller than Nn in natural images, we employ the second metric called the balance error rate (BER) to obtain a more balanced evaluation by equally considering the shadow and non-shadow regions:

![](http://latex.codecogs.com/gif.latex?\\BER=\left( 1-\frac{1}{2}\left( \frac{TP}{N_p}+\frac{TN}{N_n} \right) \right))

Note that unlike the accuracy metric, for BER, the lower its value, the better the detection result is.

